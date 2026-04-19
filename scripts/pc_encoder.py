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

class TokenProjector(nn.Module):
    def __init__(self, d_model: int, out_dim: int = 1024, num_tokens: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        # Shared basis from view query -> K token slots in hidden space
        self.base = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_tokens * hidden_dim),
        )

        # Normalize the base tokens before modulation
        self.norm_base = nn.LayerNorm(hidden_dim)

        # Camera-conditioned FiLM modulation
        self.to_gamma_beta = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2 * hidden_dim),
        )

        # Shared projection hidden -> MVDream token dim
        self.out = nn.Linear(hidden_dim, out_dim)

        # --- Zero Initialization ---
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, view_queries: torch.Tensor, cam_feats: torch.Tensor) -> torch.Tensor:
        """
        view_queries: (B, V, D)
        cam_feats:    (B, V, D)
        returns:      (B, V, K, 1024)
        """
        B, V, D = view_queries.shape

        # 1. Generate base tokens
        base = self.base(view_queries).view(B, V, self.num_tokens, self.hidden_dim)  # (B,V,K,H)

        # 2. Normalize base tokens (crucial for stable FiLM)
        base = self.norm_base(base)

        # 3. Generate modulation params
        gamma_beta = self.to_gamma_beta(cam_feats)  # (B,V,2H)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)  # each (B,V,H)

        # 4. Apply FiLM (broadcast over token axis K)
        # mod = base * (1 + gamma) + beta
        mod = base * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)

        # 5. Project to target dimension
        tokens = self.out(mod)  # (B,V,K,1024)

        return tokens

class PointCloudAttnBlock(nn.Module):
    def __init__(self, d_model=576, n_heads=8, num_views=4, device="cuda", dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_views = num_views

        # Local Self-Attention using standard PyTorch MultiheadAttention
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.norm3 = nn.LayerNorm(d_model).to(device)
        self.norm4 = nn.LayerNorm(d_model).to(device)
        self.norm5 = nn.LayerNorm(d_model).to(device)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True).to(device)
        self.cam_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True).to(device)

        # Global Cross-Attention connecting points to the memory bank
        self.global_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True).to(device)

        # Cross-attention to extract features from the dense 3D point cloud
        self.point_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True).to(device)

        self.norm_m1 = nn.LayerNorm(d_model).to(device)
        self.norm_m2 = nn.LayerNorm(d_model).to(device)
        self.mem_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True).to(device)

        hidden_dim = int(d_model * 1.5)
        self.mlp_self_attn = SwiGLU(d_model, hidden_dim).to(device)
        self.mlp_mem = SwiGLU(d_model, hidden_dim).to(device)
        self.mlp_cross_attn = SwiGLU(d_model, hidden_dim).to(device)

    def forward(self, x_dense, mask, cam_features, view_queries, m_tokens):
        residual_x = x_dense
        x_dense = self.norm1(x_dense)
        attn_out,_ = self.self_attn(
            x_dense, x_dense, x_dense, key_padding_mask=~mask, need_weights=False
        )
        x_dense = residual_x + attn_out

        residual = x_dense
        # Points check memory
        attn_out,_ = self.global_cross(
            query=self.norm2(x_dense), key=m_tokens, value=m_tokens, need_weights=False
        )

        x_dense = residual + attn_out
        x_dense = x_dense + self.mlp_self_attn(self.norm3(x_dense))

        # Memory Update
        residual_m = m_tokens
        m_out,_ = self.mem_cross(
            query=self.norm_m1(m_tokens),
            key=x_dense,
            value=x_dense,
            key_padding_mask=~mask,
            need_weights=False
        )
        updated_m_tokens = residual_m + m_out
        updated_m_tokens = updated_m_tokens + self.mlp_mem(self.norm_m2(updated_m_tokens))

        # View/Camera Cross-Attention
        residual_v = view_queries

        # Condition on cameras
        v_cond,_ = self.cam_cross(self.norm4(view_queries), cam_features, cam_features, need_weights=False)

        # Conditioned queries for refined points (K, V)
        v_out,_ = self.point_cross(
            query=v_cond,
            key=x_dense,
            value=x_dense,
            key_padding_mask=~mask,
            need_weights=False
        )

        # Update view_queries for the next block
        updated_view_queries = residual_v + v_out
        updated_view_queries = updated_view_queries + self.mlp_cross_attn(self.norm5(updated_view_queries))

        return x_dense, updated_view_queries, updated_m_tokens

@torch.compiler.disable
def _make_dense(x, batch_idx):
    # torch.compile() doesn't like this
    return to_dense_batch(x, batch_idx)

class PointCloudTransformer(nn.Module):
    def __init__(self,
                 d_model=576, #576, 256 + proj_down
                 n_heads=8, #8, 4
                 num_views=4,
                 m_tokens=32,
                 n_layers=4,
                 mvdream_context_size=1024,
                 num_context_tokens=4,
                 utonia_feat_dim=576,
                 cam_enc_hidden=576,
                 device='cuda',
                 return_starting_latent=False,
                 use_slot_attention=False,):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_views = num_views
        self.memory_tokens = nn.Parameter(torch.randn(1, m_tokens, d_model)).to(device)
        self.num_context_tokens = num_context_tokens

        #self.proj_down = nn.Sequential(nn.Linear(576, 512), nn.GELU(), nn.Linear(512, d_model)).to(device)

        self.blocks = nn.ModuleList([
            PointCloudAttnBlock(d_model, n_heads).to(device) for _ in range(n_layers)
        ])

        self.view_queries = nn.Parameter(torch.randn(1, 4, d_model)).to(device)
        self.cam_encoder = nn.Sequential(
            nn.Linear(16, cam_enc_hidden), nn.LayerNorm(cam_enc_hidden), nn.SiLU(), nn.Linear(cam_enc_hidden, d_model)
        ).to(device)
        # Final output heads

        self.semantic_head = TokenProjector(
            d_model=d_model,
            out_dim=mvdream_context_size,
            num_tokens=self.num_context_tokens,
            hidden_dim=256,
        ).to(device)

    def forward(self, x, cameras, batch_idx):
        # Cameras should not be flat
        assert cameras.ndim == 3
        B = cameras.shape[0]

        #x = self.proj_down(x)
        x, mask = _make_dense(x, batch_idx)
        #print(B, x_dense.shape, mask.shape)

        # View queries (v) will become the 4 condition tokens
        m = self.memory_tokens.expand(B, -1, -1)
        v = self.view_queries.expand(B, -1, -1)

        cam_feats = self.cam_encoder(cameras)

        for block in self.blocks:
            x, v, m = block(x, mask,cam_feats, v, m)

        # (B, V, K, 1024)
        mvdream_tokens = self.semantic_head(v, cam_feats)

        # (B*V, K, 1024)
        mvdream_tokens = mvdream_tokens.view(B * self.num_views, self.num_context_tokens, -1)

        latent = torch.randn_like(mvdream_tokens)

        return mvdream_tokens, latent

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
            device="cuda",
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
        self.to(device)

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
            device: str = "cuda",
            scale: float = 1.0,
    ):
        super().__init__()
        self.device = device

        self.encoder = utonia.model.load("utonia", repo_id="Pointcept/Utonia").to(device)
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