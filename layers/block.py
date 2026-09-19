# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List

import torch
from torch import Tensor, nn

from .attention import SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for injecting histogram features into ViT tokens.
    Query: main tokens (cls + patch tokens) - with spatial position info
    Key/Value: histogram tokens - NO position encoding needed (global statistics)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        init_values = None,
        device = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Query from main tokens, Key/Value from histogram tokens
        self.q = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias, device=device)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, hist_tokens):
        """
        Args:
            x: main tokens (B, N_main, D) - cls + patch tokens
            hist_tokens: histogram tokens (B, N_hist, D) - no spatial position
        Returns:
            x: updated main tokens (B, N_main, D)
        """
        B, N_main, D = x.shape
        _, N_hist, _ = hist_tokens.shape

        # Query from main tokens
        q = self.q(x).reshape(B, N_main, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        # Key and Value from histogram tokens
        kv = self.kv(hist_tokens).reshape(B, N_hist, 2, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Cross-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N_main, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class SelfAttentionBlockWithCross(nn.Module):
    """
    Self-attention block with cross-attention for histogram features.

    Architecture:
        x -> [norm1 -> self-attn -> ls1] -> (+)
          -> [norm_cross -> cross-attn -> ls_cross] -> (+)  (if enabled)
          -> [norm2 -> ffn -> ls2] -> (+) -> out
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        use_cross_attn: bool = False,  # Whether to use patch-hist cross attention
        device=None,
    ) -> None:
        super().__init__()

        self.use_cross_attn = use_cross_attn

        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        if use_cross_attn:
            self.norm_cross = norm_layer(dim)
            self.cross_attn = CrossAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
                init_values=init_values,
                device=device,
            )
            self.ls_cross = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        # FFN components
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

    def _forward_list(self, x_list: List[Tensor], rope_list=None, hist_tokens_list=None) -> List[Tensor]:
        x_out = []
        for i, (x, rope) in enumerate(zip(x_list, rope_list)):
            # Self-attention
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))

            # Cross-attention (if enabled)
            if self.use_cross_attn and hist_tokens_list is not None:
                x_attn = x_attn + self.ls_cross(
                    self.cross_attn(self.norm_cross(x_attn), hist_tokens_list[i]))

            # FFN
            x_out.append(x_attn + self.ls2(self.mlp(self.norm2(x_attn))))

        return x_out

    def forward(self, x_or_x_list, rope_or_rope_list=None, hist_tokens_or_list=None):
        """
        Main forward function supporting both single tensor and list inputs.

        Args:
            x_or_x_list: Single tensor or list of tensors
            rope_or_rope_list: RoPE embeddings (single or list)
            hist_tokens_or_list: Histogram tokens for cross-attention (single or list, optional)

        Returns:
            Single tensor or list of tensors (matches the input type)
        """
        if isinstance(x_or_x_list, Tensor):
            hist_list = [hist_tokens_or_list] if hist_tokens_or_list is not None else None
            return self._forward_list(
                [x_or_x_list],
                rope_list=[rope_or_rope_list],
                hist_tokens_list=hist_list,
            )[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _ in x_or_x_list]
            if hist_tokens_or_list is None:
                hist_tokens_or_list = [None for _ in x_or_x_list]

            return self._forward_list(
                x_or_x_list,
                rope_list=rope_or_rope_list,
                hist_tokens_list=hist_tokens_or_list,
            )
        else:
            raise AssertionError("Input must be Tensor or List[Tensor]")
