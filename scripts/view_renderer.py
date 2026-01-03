import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer,
    FoVPerspectiveCameras,
    look_at_view_transform
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader, # Or HardFlatShader for a matte look
    AmbientLights,
    TexturesVertex
)
from pytorch3d.renderer.cameras import look_at_rotation

class PointRenderer:
    def __init__(self, device='cuda', image_size=256, radius=0.015):
        """
        Args:
            radius: Slightly larger radius to make points look more 'solid'
        """
        self.device = device
        self.image_size = image_size

        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=radius,
            points_per_pixel=8
        )

        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=(1.0, 1.0, 1.0))
        ).to(device)

    @torch.no_grad()
    def render_mvdream_views(self, points, dist=2.5, elev=15):
        """
        Takes raw points and returns a batch of 4 orthogonal views.

        Args:
            points: (B, N, 3) or (N, 3) tensor of point coordinates.
            dist: Camera distance.
            elev: Camera elevation.

        Returns:
            (B*4, 3, H, W) Tensor normalized to [-1, 1] ready for MV-Dream.
        """
        # single input vs batch
        if points.ndim == 2:
            points = points.unsqueeze(0)
        B, N, _ = points.shape

        # bright pink / magenta (RGB in [0,1])
        pink = torch.tensor([1.0, 0.2, 0.8], device=self.device)  # tweak as you like

        features = pink.view(1, 1, 3).expand(B, N, 3).contiguous()
        point_cloud = Pointclouds(points=points, features=features)

        # extend to 4 views per object
        point_cloud_expanded = point_cloud.extend(4)  # (B*4)

        # 4 orthogonal azimuths per object
        azims = torch.tensor([0, 90, 180, 270], dtype=torch.float32, device=self.device)
        azims = azims.repeat(B)

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azims)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        # renderer output: (B*4, H, W, 3) in [0, 1]
        rendered_rgb = self.renderer(point_cloud_expanded, cameras=cameras)

        # (N_views, 3, H, W) in [0,1]
        out_tensor = rendered_rgb.permute(0, 3, 1, 2)

        # normalize to [-1, 1] for MVDream
        out_tensor = out_tensor * 2.0 - 1.0

        return out_tensor


class MeshRendererMVDream:
    def __init__(self, device='cuda', image_size=256, fov_deg=50.0):
        self.device = device
        self.image_size = image_size
        self.fov_deg = fov_deg
        # Rasterization settings for meshes
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0, 
            faces_per_pixel=1,
        )

        # Simple ambient lighting to keep the color uniform (like your PC version)
        # Without lights, the mesh will appear black.
        lights = AmbientLights(device=device, ambient_color=((1.0, 1.0, 1.0),))

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, lights=lights)).to(device)


    def _mvdream_to_pytorch3d_cameras(self, camera: torch.Tensor, dist_scale=2.5, world_up=(0, 1, 0)):
        if camera.ndim == 2 and camera.shape[1] == 16:
            c2w = camera.view(-1, 4, 4)
        elif camera.ndim == 3 and camera.shape[-2:] == (4, 4):
            c2w = camera
        else:
            raise ValueError(f"Bad camera shape: {camera.shape}")

        c2w = c2w.to(self.device).float()

        # camera centers on orbit
        C = c2w[:, :3, 3] * dist_scale
        C.to(self.device)

        up = torch.tensor(world_up, device=self.device, dtype=torch.float32).view(1, 3).expand(C.shape[0], 3).to(self.device)

        # world->cam rotation: look from C to origin with fixed up
        R = look_at_rotation(C, at=((0, 0, 0),), up=up).to(self.device)  # (V,3,3)

        # PyTorch3D: X_cam = X_world @ R^T + T
        T = -torch.bmm(R, C[:, :, None])[:, :, 0].to(self.device)

        return FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self.fov_deg, znear=0.01, zfar=10.0)

    @torch.no_grad()
    def render_mvdream_views(self, verts, faces, camera, dist_scale=2.5):
        if verts.ndim == 2:
            verts = verts.unsqueeze(0)   # (1, N, 3)
        if faces.ndim == 2:
            faces = faces.unsqueeze(0)   # (1, F, 3)

        B, Nverts, _ = verts.shape
        _, Nfaces, _ = faces.shape

        # sanity
        assert faces.max() < Nverts, (faces.max().item(), Nverts)

        pink = torch.tensor([1.0, 0.2, 0.8], device=self.device)
        verts_rgb = pink.view(1, 1, 3).expand(B, Nverts, 3).contiguous()
        textures = TexturesVertex(verts_features=verts_rgb)

        mesh = Meshes(verts=verts, faces=faces, textures=textures)

        cameras = self._mvdream_to_pytorch3d_cameras(camera, dist_scale=dist_scale)
        Vviews = cameras.R.shape[0]

        mesh_expanded = mesh.extend(Vviews)

        rendered = self.renderer(mesh_expanded, cameras=cameras)
        alpha = rendered[..., 3]
        print("alpha max:", alpha.max().item())  # keep until you see >0

        rgb = rendered[..., :3]
        out = rgb.permute(0, 3, 1, 2) * 2.0 - 1.0
        return out