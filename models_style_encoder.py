"""Photographic style encoder.

ViT-L/16 backbone whose tokens are conditioned on LAB histogram tokens through
cross-attention, followed by a projector/predictor head that produces the
photographic embedding `z` used for style transfer.

The ViT building blocks in `layers/` are derived from DINOv3
(see LICENSE-DINOv3.md).
"""

import logging
from functools import partial
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn.init
from torch import Tensor, nn
import torch.nn.functional as F

from layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlockWithCross, SwiGLUFFN
from layers.utils import named_apply


logger = logging.getLogger("PhotographicStyleEncoder")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class AttentionPooling(nn.Module):
    # Adaptive attention pooling for patch tokens
    def __init__(self, dim, init_std=1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scorer = nn.Linear(dim, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))  # log temperature
        # init
        nn.init.normal_(self.scorer.weight, std=init_std)
        nn.init.constant_(self.scorer.bias, 0.)

    def forward(self, patch_tokens):  # -> tuple[Any, Tensor]:
        # patch_tokens: (B, N, D)
        x = self.norm(patch_tokens)
        logits = self.scorer(x).squeeze(-1)           # (B, N)
        logits = logits / (torch.exp(self.log_temp) + 1e-8)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B, N, 1)
        pooled = (weights * patch_tokens).sum(dim=1)         # (B, D)
        return pooled, weights.squeeze(-1)  # return weights for visualization


class HistTokenProjector(nn.Module):
    """Per-layer linear projections for the L and AB histogram tokens."""

    def __init__(self, emb_vit, cross_attention_layers):
        super().__init__()
        self.cross_attention_layers = cross_attention_layers
        self.hist_projections = nn.ModuleDict({
            str(layer_idx): nn.ModuleDict({
                "L": nn.Linear(emb_vit, emb_vit),
                "AB": nn.Linear(emb_vit, emb_vit),
            })
            for layer_idx in cross_attention_layers
        })

    def forward(self, hist_tokens):
        """
        hist_tokens: (B, 2, emb_vit)
        return: dict[layer_idx] -> projected hist_tokens (B, 2, emb_vit)
        """
        proj_tokens = {}
        for layer_idx in self.cross_attention_layers:
            projection = self.hist_projections[str(layer_idx)]
            L_proj = projection["L"](hist_tokens[:, 0:1, :])
            AB_proj = projection["AB"](hist_tokens[:, 1:2, :])
            proj_tokens[layer_idx] = torch.cat([L_proj, AB_proj], dim=1)
        return proj_tokens


class StyleBranch(nn.Module):
    def __init__(
            self,
            vit_args: Dict[str, Any],
            # parameters for histogram encoder
            hist_h: int = 128,
            cnn_base_channels: int = 64,
            cnn_num_blocks: int = 3,
            # parameters for projector/predictor
            projector_hidden_dim: int = 512,
            projector_output_dim: int = 1024,
            predictor_hidden_dim: int = 512,
            predictor_output_dim: int = 1024,
        ) -> None:
        super().__init__()
        self.vit = DinoVisionTransformer(**vit_args)
        self.vit.init_weights()     # Initialize weights of ViT

        self.histogram_encoder = HistogramEncoder(
            hist_h=hist_h,
            img_size=vit_args['img_size'],
            embed_dim=vit_args['embed_dim'],
            base_channels=cnn_base_channels,
            num_blocks=cnn_num_blocks,
        )

        self.embed_dim = self.vit.embed_dim
        self.num_patches = (self.vit.img_size // self.vit.patch_size) ** 2

        projector_input_dim = self.embed_dim * 2
        self.pool_attn = AttentionPooling(self.embed_dim)

        self.projector = nn.Sequential(
            nn.Linear(projector_input_dim, projector_hidden_dim, bias=False), nn.BatchNorm1d(projector_hidden_dim), nn.GELU(approximate='tanh'),
            nn.Linear(projector_hidden_dim, projector_hidden_dim, bias=False), nn.GELU(approximate='tanh'),
            nn.Linear(projector_hidden_dim, projector_output_dim, bias=False), nn.BatchNorm1d(projector_output_dim, affine=False)
        )

        self.predictor = nn.Sequential(
            nn.Linear(projector_output_dim, predictor_hidden_dim, bias=False),
            nn.BatchNorm1d(predictor_hidden_dim),
            nn.ReLU(),  # hidden layer
            nn.Linear(predictor_hidden_dim, predictor_output_dim),
        )

    def aggregate_tokens(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Aggregate the cls token and the attention-pooled patch tokens.

        Args:
            features_dict: Dictionary containing:
                - x_norm_clstoken: (B, embed_dim)
                - x_norm_patchtokens: (B, num_patches, embed_dim)

        Returns:
            aggregated: (B, projector_input_dim)
        """
        cls_token = features_dict['x_norm_clstoken']  # (B, D)
        patch_tokens = features_dict['x_norm_patchtokens']  # (B, num_patches, D)
        patch_pooled, _ = self.pool_attn(patch_tokens)      # (B, D)
        return torch.cat([cls_token, patch_pooled], dim=1)  # (B, 2D)

    def forward(self, x):
        hist_token, ab_hist, l_hist = self.histogram_encoder(x)
        features_dict = self.vit.forward_features(x, hist=hist_token)
        aggregated = self.aggregate_tokens(features_dict)
        projected = self.projector(aggregated)
        predicted = self.predictor(projected)

        return dict(
            features_dict=features_dict,
            aggregated=aggregated,
            z=projected,
            p=predicted,
            ab_hist=ab_hist,        # (B,1,h,h)
            l_hist=l_hist,          # (B,1,h)
        )


class LabHistBlock(nn.Module):
    """Computes the 2D ab-histogram feature of a LAB image in [0, 1].

    The inverse-quadratic kernel follows HistoGAN:
        Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN:
        Controlling Colors of GAN-Generated and Real Images via Color
        Histograms." In CVPR, 2021.
    """

    def __init__(self, h=64, insz=512, sigma=0.02):
        super().__init__()
        self.h = h
        self.insz = insz
        self.sigma = sigma

        self.register_buffer('bins', torch.tensor(np.linspace(0, 1, num=self.h), dtype=torch.float32))

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            x = F.interpolate(x, size=(self.insz, self.insz), mode='bilinear', align_corners=False)

        L = x.shape[0]  # size of mini-batch
        if x.shape[1] > 3:
            x = x[:, :3, :, :]
        X = torch.unbind(x, dim=0)
        hists = torch.zeros((x.shape[0], 1, self.h, self.h), dtype=torch.float32, device=x.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))

            Ia = torch.unsqueeze(I[:, 1], dim=1)
            Ib = torch.unsqueeze(I[:, 2], dim=1)

            diff_a = abs(Ia - self.bins.unsqueeze(0))
            diff_b = abs(Ib - self.bins.unsqueeze(0))

            diff_a = 1 / (1 + torch.pow(torch.reshape(diff_a, (-1, self.h)), 2) / self.sigma ** 2)
            diff_b = 1 / (1 + torch.pow(torch.reshape(diff_b, (-1, self.h)), 2) / self.sigma ** 2)

            diff_a = diff_a.type(torch.float32)
            diff_b = diff_b.type(torch.float32)

            hists[l, 0, :, :] = torch.mm(torch.t(diff_a), diff_b)

        # normalization
        hists_normalized = hists / (
            ((hists.mean(dim=1)).mean(dim=1)).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)

        return hists_normalized


class LHistBlock(nn.Module):
    """Computes the 1D luminance histogram feature of a LAB image in [0, 1]."""

    def __init__(self, h=64, insz=512, sigma=0.02):
        super().__init__()
        self.h = h
        self.insz = insz
        self.sigma = sigma

        self.register_buffer('bins', torch.tensor(np.linspace(0, 1, num=self.h), dtype=torch.float32))

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        B, C, H, W = x.shape

        if H > self.insz or W > self.insz:
            x = F.interpolate(x, size=(self.insz, self.insz), mode='bilinear', align_corners=False)

        if x.shape[1] > 1:       # get the L channel
            L = x[:, 0:1, :, :]  # shape (B, 1, H', W')
        else:
            L = x

        L_flat = L.view(B, -1)  # (B, N_pixels)

        hists = []
        for b in range(B):
            diff = torch.abs(L_flat[b:b+1].T - self.bins)  # shape (N_pixels, h)
            hist = (1 / (1 + diff ** 2 / self.sigma ** 2)).mean(dim=0)
            hist = hist / (hist.mean() + 1e-6)
            hists.append(hist)

        return torch.stack(hists, dim=0).unsqueeze(1)  # (B, 1, h)


class CNNHistogramEncoder(nn.Module):
    """Encodes the 2D ab-histogram and the 1D luminance histogram into tokens."""

    def __init__(
        self,
        hist_h: int = 64,
        embed_dim: int = 768,
        base_channels: int = 64,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.hist_h = hist_h
        self.embed_dim = embed_dim

        self.l_encoder = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.GELU(approximate='tanh'),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU(approximate='tanh'),
            nn.MaxPool1d(2),

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.GELU(approximate='tanh'),
            nn.AdaptiveAvgPool1d(1),
        )
        self.l_proj = nn.Linear(base_channels * 4, embed_dim)

        self.ab_encoder, self.ab_out_channels = self._build_2d_encoder(base_channels, num_blocks)
        self.ab_proj = nn.Linear(self.ab_out_channels, embed_dim)

    def _build_2d_encoder(self, base_channels, num_blocks):
        """Build 2D CNN encoder with progressive downsampling."""
        layers = []
        in_channels = 1

        for i in range(1, num_blocks + 1):
            out_channels = base_channels * (2 ** i)

            # Conv block
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(approximate='tanh'),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(approximate='tanh'),
            ])

            # Downsample (except last block)
            if i < num_blocks - 1:
                layers.append(nn.MaxPool2d(2))

            in_channels = out_channels
            last_channels = out_channels

        # Global pooling
        layers.append(nn.AdaptiveAvgPool2d(1))

        return nn.Sequential(*layers), last_channels

    def forward(self, hists_ab, hists_L):
        """
        Args:
            hists_ab: (B, 1, h, h)
            hists_L: (B, 1, h) or (B, h)
        Returns:
            tokens: (B, 2, embed_dim)
        """
        dtype = next(self.parameters()).dtype
        hists_ab = hists_ab.to(dtype=dtype)
        hists_L = hists_L.to(dtype=dtype)

        # Process L
        if hists_L.dim() == 3:
            l_input = hists_L  # (B, 1, h)
        else:
            l_input = hists_L.unsqueeze(1)  # (B, 1, h)

        l_features = self.l_encoder(l_input).squeeze(-1)  # (B, C)
        l_token = self.l_proj(l_features).unsqueeze(1)  # (B, 1, embed_dim)

        # Process AB
        ab_features = self.ab_encoder(hists_ab).squeeze(-1).squeeze(-1)  # (B, C)
        ab_token = self.ab_proj(ab_features).unsqueeze(1)  # (B, 1, embed_dim)

        tokens = torch.cat([l_token, ab_token], dim=1)  # (B, 2, embed_dim)
        return tokens


class HistogramEncoder(nn.Module):
    def __init__(self, hist_h: int, img_size: int, embed_dim: int,
                 base_channels: int = 64, num_blocks: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hist_h = hist_h
        self.img_size = img_size

        self.ab_hist_block = LabHistBlock(h=hist_h, insz=img_size)
        self.l_hist_block = LHistBlock(h=hist_h, insz=img_size)
        self.hist_token_embed = CNNHistogramEncoder(
            hist_h=hist_h,
            embed_dim=embed_dim,
            base_channels=base_channels,
            num_blocks=num_blocks,
        )

    def forward(self, x):
        # x is already in [0,1] range LAB format
        ab_hist = self.ab_hist_block(x)     # (B,1,h,h)
        l_hist = self.l_hist_block(x)       # (B,1,h)

        return self.hist_token_embed(ab_hist, l_hist), ab_hist, l_hist


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        device: Any | None = None,
    ):
        super().__init__()

        norm_layer_cls = norm_layer_dict[norm_layer]
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size

        self.cross_attention_layers = list(range(1, depth - 1, 3))
        self.histogram_token_projections = HistTokenProjector(
            emb_vit=embed_dim, cross_attention_layers=self.cross_attention_layers)

        logger.info(f"Using cross-attention with histogram tokens at layers: {self.cross_attention_layers}")

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))

        logger.info(f"using base={pos_embed_rope_base} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )

        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth

        blocks_list = [
            SelfAttentionBlockWithCross(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                device=device,
                use_cross_attn=(i in self.cross_attention_layers),
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer_cls(embed_dim)

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        named_apply(init_weights_vit, self)

    def prepare_tokens(self, x: Tensor) -> Tuple[Tensor, Tuple[int]]:
        """Prepare tokens, in the order [cls_token, patch tokens]."""
        x_patches = self.patch_embed(x)
        B, H, W, _ = x_patches.shape
        x_patches = x_patches.flatten(1, 2)

        x = torch.cat([self.cls_token.expand(B, -1, -1), x_patches], dim=1)
        return x, (H, W)

    def forward_features(self, x: Tensor, hist: Tensor) -> Dict[str, Tensor]:
        x, (H, W) = self.prepare_tokens(x)
        # RoPE is only applied to patch tokens, not to the cls/histogram tokens
        rope_sincos = self.rope_embed(H=H, W=W)

        hist_tokens = self.histogram_token_projections(hist)

        for block_idx, blk in enumerate(self.blocks):
            if block_idx in self.cross_attention_layers:
                x = blk(x, rope_sincos, hist_tokens[block_idx])
            else:
                x = blk(x, rope_sincos)

        x_norm = self.norm(x)

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
        }


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()
