import torch
import torch.nn as nn


class LUT(nn.Module):
    """AdaIN-style LUT transfer model.

    The content image is concatenated with normalized pixel coordinates, then
    rescaled with the style/content statistics predicted from the style and
    content embeddings.
    """

    def __init__(self, dim=512, emb_in_dim=1024):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(5, 64, (1, 1), 1),
            nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv2d(64, 128, (1, 1), 1),
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, dim, (1, 1), 1),
            nn.GroupNorm(16, dim), nn.GELU(),
        )
        self.to_mean = nn.Sequential(
            nn.Linear(emb_in_dim, dim),
            nn.GroupNorm(16, dim),
            nn.Linear(dim, dim),
        )
        self.to_std = nn.Sequential(
            nn.Linear(emb_in_dim, dim),
            nn.GroupNorm(16, dim),
            nn.Linear(dim, dim), nn.ELU(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(dim, 128, (1, 1), 1),
            nn.GELU(),
            nn.Conv2d(128, 64, (1, 1), 1),
            nn.GELU(), nn.GroupNorm(16, 64),
            nn.Conv2d(64, 3, (1, 1), 1),
        )

    def forward(self, content, style, content_emb, guidance_scale=1.0):
        delta_x = torch.linspace(-1, 1, content.shape[-2],
                                 device=content.device, dtype=content.dtype
                                 ).reshape(1, 1, -1, 1).repeat(content.shape[0], 1, 1, content.shape[-1])
        delta_y = torch.linspace(-1, 1, content.shape[-1],
                                 device=content.device, dtype=content.dtype
                                 ).reshape(1, 1, 1, -1).repeat(content.shape[0], 1, content.shape[-2], 1)
        latent = self.up(torch.cat([content, delta_x, delta_y], 1))

        latent_std = torch.std(latent, dim=(2, 3), keepdim=False)
        latent_mean = torch.mean(latent, dim=(2, 3), keepdim=False)
        latent_rescaled = (latent - latent_mean.detach()[..., None, None]) / \
                          torch.clamp(latent_std.detach()[..., None, None], 1e-7, 1e7)

        mean, mean_content = self.to_mean(torch.cat([style, content_emb], dim=0)).chunk(2)
        std, std_content = self.to_std(torch.cat([style, content_emb], dim=0)).chunk(2)

        res = (latent_rescaled * (latent_std.detach() + std - std_content)[..., None, None]
               + (latent_mean.detach() + (mean - mean_content))[..., None, None])

        if guidance_scale != 1:
            input_identity, res = self.out(torch.cat([latent, res], 0)).chunk(2, 0)
            res = res * guidance_scale + (1 - guidance_scale) * input_identity
        else:
            res = self.out(res)

        return res
