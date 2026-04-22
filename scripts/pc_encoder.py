import torch
import torch.nn as nn
import numpy as np
import utonia
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

class SwiGLU(nn.Module):
    """
    More parameter-efficient than standard GELU MLPs.
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class PointCloudCrossAttnBlock(nn.Module):
    """Asymmetric cross-attention: queries at latent_dim, keys/values at utonia_feat_dim."""
    def __init__(self, latent_dim=256, utonia_feat_dim=576, n_heads=4, dropout=0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(utonia_feat_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            kdim=utonia_feat_dim,
            vdim=utonia_feat_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(latent_dim)
        self.ff = SwiGLU(latent_dim, int(latent_dim * 1.5))

    def forward(self, latent_tokens, pc_feat, mask):
        # mask: (B, N) bool, True = valid point
        x, _ = self.attn(
            self.norm_q(latent_tokens),
            self.norm_kv(pc_feat),
            pc_feat,
            key_padding_mask=~mask,
            need_weights=False,
        )
        latent_tokens = latent_tokens + x
        latent_tokens = latent_tokens + self.ff(self.norm_ff(latent_tokens))
        return latent_tokens


class PointCloudSelfAttnBlock(nn.Module):
    """Cheap self-attention over all latent tokens (16 tokens @ 256-dim)."""
    def __init__(self, latent_dim=256, n_heads=4, dropout=0.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(latent_dim)
        self.attn   = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(latent_dim)
        self.ff     = SwiGLU(latent_dim, int(latent_dim * 1.5))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), x, x, need_weights=False)[0]
        x = x + self.ff(self.norm_ff(x))
        return x

class PointCloudTransformerSmall(nn.Module):
    def __init__(
            self,
            n_heads=4,
            num_views=4,
            num_tokens=4,
            n_self_attn_layers=4,
            latent_dim=256,
            mvdream_context_size=1024,
            utonia_feat_dim=576,
    ):
        super().__init__()
        self.num_views  = num_views
        self.num_tokens = num_tokens
        total_tokens    = num_views * num_tokens  # 16

        self.camera_encoder = nn.Sequential(
            nn.Linear(16, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Learnable latent tokens + view-index embeddings for consistency
        self.latent_tokens = nn.Parameter(torch.randn(1, total_tokens, latent_dim))

        # 1 cross-attention block (reads point cloud)
        self.cross_attn = PointCloudCrossAttnBlock(latent_dim, utonia_feat_dim, n_heads)

        # N self-attention blocks (view consistency + refinement)
        self.self_attn_blocks = nn.ModuleList([
            PointCloudSelfAttnBlock(latent_dim, n_heads)
            for _ in range(n_self_attn_layers)
        ])

        self.up_proj = nn.Linear(latent_dim, mvdream_context_size)

    def forward(self, pc_feat, mask, cameras):
        B = pc_feat.shape[0]
        cam_embed = self.camera_encoder(cameras)                          # (B, 4, latent_dim)
        cam_embed = cam_embed.repeat_interleave(self.num_tokens, dim=1)  # (B, 16, latent_dim)

        latent_tokens = self.latent_tokens.expand(B, -1, -1) + cam_embed

        # Cross-attend to point cloud once
        latent_tokens = self.cross_attn(latent_tokens, pc_feat, mask)

        # Refine + enforce view consistency
        for block in self.self_attn_blocks:
            latent_tokens = block(latent_tokens)

        # Project to MVDream context dim and reshape for multi-view conditioning
        out = self.up_proj(latent_tokens)                             # (B, 16, 1024)
        out = out.view(B * self.num_views, self.num_tokens, -1)       # (B*4, 4, 1024)

        return out, None

class PointCloudEncoder(nn.Module):
    """
    Encodes a fixed-size batch of point clouds into latents for flow matching.

    Assumes all point clouds in the batch have the same number of points N,
    so coords is always (B, N, 3).

    Args:
        freeze_encoder (bool): Freeze Utonia weights; only train the projector.
        device (str):       Target device.
    """

    def __init__(
            self,
            freeze_encoder: bool = True,
            scale: float = 1.0,
    ):
        super().__init__()

        self.encoder = utonia.model.load("utonia", repo_id="Pointcept/Utonia")
        self.encoder.eval()
        self.scale = scale

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.config = [
            dict(type="NormalizeCoord"),
            dict(type="RandomScale", scale=[self.scale, self.scale]),
            #dict(type="CenterShift", apply_z=True), The points shouldn't be moved to the center as they represent an incomplete object
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
        self.transform = utonia.transform.Compose(self.config)

    def encode(
            self,
            coords: torch.Tensor,           # (Total_N, 3)
            batch_idx: torch.Tensor = None,        # (Total_N,)
            colors: torch.Tensor = None,    # (Total_N, 3)
            normals: torch.Tensor = None,   # (Total_N, 3)
            batched: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Encode a batch of flattened point clouds into flow-matching latents.
        """

        points_flat = coords.cpu().numpy()

        if normals is not None:
            normals_flat = normals.cpu().numpy()
        else:
            normals_flat = np.zeros_like(points_flat, dtype=np.float32)

        if colors is not None:
            colors_flat = colors.cpu().numpy()
        else:
            colors_flat = np.zeros_like(points_flat, dtype=np.float32)

        # Utonia Transforms + Feature Extraction
        with torch.no_grad():
            if batched:
                point_list = []
                assert batch_idx is not None
                batch = batch_idx.cpu().numpy()
                B = int(torch.max(batch_idx).item() + 1)
                for b in range(B):
                    mask = (batch == b)

                    point = {
                        "coord": points_flat[mask],
                        "color": colors_flat[mask],
                        "normal": normals_flat[mask],
                    }

                    point = self.transform(point)
                    point_list.append(point)

                points = utonia.data.collate_fn(point_list)
            else:
                point = {
                    "coord": points_flat,
                    "color": colors_flat,
                    "normal": normals_flat,
                }
                points = self.transform(point)

            for key in points.keys():
                if isinstance(points[key], torch.Tensor):
                    points[key] = points[key].cuda(non_blocking=True)

            points = self.encoder(points)

        for _ in range(0):
            assert "pooling_parent" in points.keys()
            assert "pooling_inverse" in points.keys()
            parent = points.pop("pooling_parent")
            inverse = points.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, points.feat[inverse]], dim=-1)
            points = parent

        pc_features = points.feat

        return pc_features, points.batch if batched else None

    def forward(
            self,
            coords: torch.Tensor,
            batch_idx: torch.Tensor = None,
            colors: torch.Tensor | None = None,
            normals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.encode(coords, colors=colors, normals=normals, batch_idx=batch_idx)