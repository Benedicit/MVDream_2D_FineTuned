import torch
import torch.nn as nn
import numpy as np
import torch_scatter

import utonia


class PointCloudProjector(nn.Module):
    """
    Learned Decoder that maps a pooled Utonia feature vector (B, enc_dim)
    to a spatial flow-matching latent map for multiple views (B*V, C, H, W).
    """
    def __init__(self, enc_dim: int, out_channels: int = 4, num_views: int = 4, out_size: int = 32, hidden_dim: int = 512, device="cuda"):
        super().__init__()
        self.out_channels = out_channels
        self.num_views = num_views
        self.out_size = out_size

        # Start with a small 8x8 spatial resolution
        self.base_size = 4
        self.base_channels = 256

        # 1. Map 1D feature to a 3D tensor (C, H, W)
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.base_channels * self.base_size * self.base_size)
        ).to(device)

        # 2. Convolutional Upsampling to target size (32x32)
        self.upsample = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(self.base_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            # Final projection to (num_views * out_channels)
            nn.Conv2d(32, out_channels * num_views, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=out_channels * num_views, num_channels=out_channels * num_views)
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, enc_dim) → (B * num_views, out_channels, out_size, out_size)"""
        B = x.shape[0]
        x = self.mlp(x)

        # Reshape into a spatial grid
        x = x.view(B, self.base_channels, self.base_size, self.base_size)

        # Upsample: returns (B, 16, 32, 32)
        x = self.upsample(x)

        # Reshape into individual views: (B * 4, 4, 32, 32)
        x = x.view(B * self.num_views, self.out_channels, self.out_size, self.out_size)
        return x



class PointCloudEncoder(nn.Module):
    """
    Encodes a fixed-size batch of point clouds into latents for flow matching.

    Assumes all point clouds in the batch have the same number of points N,
    so coords is always (B, N, 3).

    Args:
        pretrained (str):   HuggingFace repo id or local .pth checkpoint path.
        enc_dim (int):      Output feature dimension of the Utonia encoder.
        latent_dim (int):   Target latent dimension for the flow matching.
        hidden_dim (int):   Hidden size of the projection MLP.
        freeze_encoder (bool): Freeze Utonia weights; only train the projector.
        device (str):       Target device.
    """

    def __init__(
            self,
            enc_dim: int = 1008,
            hidden_dim: int = 512,
            freeze_encoder: bool = True,
            device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # --- Utonia / PointTransformerV3 backbone ---
        self.encoder = utonia.model.load("utonia", repo_id="Pointcept/Utonia").cuda()
        self.encoder.eval()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.scale = 1.0
        self.config = [
            dict(type="NormalizeCoord"), # uncomment when applying to objects
            dict(type="RandomScale", scale=[self.scale, self.scale]),
            dict(type="CenterShift", apply_z=True), # remove this for outdoor LiDAR and ensure the ego-vehicle is at the origin (0, 0, 0), with the road plane aligned with the xy-plane
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
        # --- Learned projection into the flow-matching latent space ---
        self.projector = PointCloudProjector(enc_dim=enc_dim, hidden_dim=hidden_dim)

        self.projector = torch.compile(self.projector, mode="max-autotune", dynamic=False, )

    @staticmethod
    def _build_input_dict(
            coords: np.ndarray,   # (B * N, 3)
            colors: np.ndarray | None,   # (B * N, 3)
            normals: np.ndarray | None,  # (B * N, 3)
            batch : np.ndarray | None
    ) -> dict:
        """
        Flatten the (B, N, *) batch into the Pointcept flat format and build
        the required input dict with coord, feat, offset, and batch keys.

        When color or normal is absent Utonia expects default zeros
        (see README: 'When color or normal is absent, please apply default
        zeros to the missing modality').
        """
        B_N, _ = coords.shape
        if colors is None:
            colors = np.zeros((B_N,3))

        if normals is None:
            normals = np.zeros((B_N,3))

        '''
        # Flat tensors: (B*N, *)
        coord_flat = coords.reshape(B * N, 3)
        colors = colors.reshape(B * N, 3)
        normals = normals.reshape(B * N, 3)
        '''

        return {
            "coord":  coords,  # (B*N, 3)
            "color":  colors,
            "normal": normals,
            "batch":  batch,   # (B*N,)
        }

    def encode(
            self,
            coords: torch.Tensor,           # (Total_N, 3)
            colors: torch.Tensor = None,   # (Total_N, 3)
            normals: torch.Tensor = None,  # (Total_N, 3)
            batch_lengths: list[int] = None, # List of lengths per batch item
    ) -> torch.Tensor:
        """
        Encode a batch of flattened point clouds into flow-matching latents.
        """
        B = len(batch_lengths)

        # Use provided flattened tensors or create defaults
        points_flat = coords.cpu().numpy()

        if normals is not None:
            normals_flat = normals.cpu().numpy()
        else:
            normals_flat = np.zeros(points_flat.shape, dtype=np.float32)

        if colors is not None:
            colors_flat = colors.cpu().numpy()
        else:
            colors_flat = np.zeros(points_flat.shape, dtype=np.float32)

        # Efficiently generate batch index on CPU using numpy
        # [0,0... 1,1... B-1, B-1...]
        batch = np.repeat(np.arange(B), batch_lengths)

        data_dict = self._build_input_dict(points_flat, colors_flat, normals_flat, batch)

        # --- Utonia forward: returns a Point object with (B*N, enc_dim) ---
        with torch.no_grad():
            point = self.transform(data_dict)
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].cuda(non_blocking=True)

            point = self.encoder(point)

        for _ in range(1):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent


        pc_features = point.feat
        new_batch_idx = point.batch
        pooled = torch_scatter.scatter_mean(pc_features, new_batch_idx, dim=0, dim_size=B)
        return self.projector(pooled)

    def forward(
            self,
            coords: torch.Tensor,
            colors: torch.Tensor | None = None,
            normals: torch.Tensor | None = None,
            batch_lengths: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.encode(coords, colors=colors, normals=normals, batch_lengths=batch_lengths)

if __name__ == '__main__':
    encoder = PointCloudEncoder(
        pretrained="Pointcept/Utonia",
        enc_dim=768,       # Utonia PTv3 output dim
        latent_dim=512,    # target flow-matching latent dim
        freeze_encoder=True,
        device="cuda",
    )