import torch
import math
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer,
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

class MVDreamRenderer:
    def __init__(self, device='cuda', image_size=256, fov_deg=50.0):
        self.device = device
        self.image_size = image_size
        self.fov_deg = fov_deg

    def _mvdream_to_pytorch3d_cameras(
        self,
        camera: torch.Tensor,
        dist_scale: float = 2.5,
        view_order: str = "frbl",  # front,right,back,left
    ):
        """
        camera: (V,16) or (V,4,4) cam2world from MVDream get_camera (blender_coord=True)
        view_order:
          - "frbl" = front, right, back, left  (default, matches MVDream az=0,90,180,270)
          - "flbr" = front, left, back, right
          - "fblr" = front, back, left, right  (etc.)
        """

        # ---- parse to (V,4,4)
        if camera.ndim == 2 and camera.shape[1] == 16:
            c2w = camera.view(-1, 4, 4)
        elif camera.ndim == 3 and camera.shape[-2:] == (4, 4):
            c2w = camera
        else:
            raise ValueError(f"Bad camera shape: {camera.shape}")

        c2w = c2w.to(self.device).float()
        V = c2w.shape[0]

        # ---- MVDream get_camera(4, az=0..270) with blender_coord=True gives:
        # az=0   -> front  (camera on -Y)
        # az=90  -> right  (camera on +X)
        # az=180 -> back   (camera on +Y)
        # az=270 -> left   (camera on -X)
        # So default returned order is already [front, right, back, left] = "frbl".
        if V >= 4:
            order_map = {
                "frbl": [0, 1, 2, 3],
                "flbr": [0, 3, 2, 1],
                "fbrl": [0, 2, 1, 3],
                "fblr": [0, 2, 3, 1],
            }
            if view_order in order_map:
                idx = torch.tensor(order_map[view_order], device=self.device, dtype=torch.long)
                c2w = c2w.index_select(0, idx)

        # ---- cam2world -> world2cam
        # The rotation columns of c2w are the camera axes expressed in world coords
        # To get w2c rotation: take the transpose (inverse for rotation matrices)
        R_c2w = c2w[:, :3, :3]  # (V,3,3)
        T_c2w = c2w[:, :3, 3] * dist_scale  # (V,3) camera center in world

        up_vectors = R_c2w[:, :, 1]
        up_list = [tuple(up_vectors[i].cpu().numpy()) for i in range(V)]

        R, T = look_at_view_transform(
            eye=T_c2w,
            up=up_list,
            device=self.device)

        return FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=self.fov_deg,
            znear=0.01,
            zfar=10.0,
        )

class PointRenderer(MVDreamRenderer):
    def __init__(self, device='cuda', image_size=256, radius=0.015):
        """
        Args:
            radius: Slightly larger radius to make points look more 'solid'
        """
        super().__init__(device, image_size)

        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=radius,
            points_per_pixel=4
        )

        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=(1.0, 1.0, 1.0))
        ).to(device)


    @torch.no_grad()
    def render_mvdream_views(self, points, camera, dist_scale=2.5):
        """
        Takes raw points and returns a batch of 4 orthogonal views.

        Args:
            points: (B, N, 3) or (N, 3) tensor of point coordinates.
            dist_scale: Camera distance.
            camera: Camera given by MVDream

        Returns:
            (B*4, 3, H, W) Tensor normalized to [-1, 1] ready for MV-Dream.
        """
        # single input vs batch
        if points.ndim == 2:
            points = points.unsqueeze(0)
        B, N, _ = points.shape

        centroid = points.mean(dim=1, keepdim=True)  # (B, 1, 3)
        points = points - centroid
        max_dist = torch.sqrt((points ** 2).sum(dim=2)).max(dim=1, keepdim=True)[0]  # (B, 1)
        points = points / max_dist.unsqueeze(2)

        # bright pink / magenta (RGB in [0,1])
        pink = torch.tensor([1.0, 0.2, 0.8], device=self.device)  # tweak as you like

        features = pink.view(1, 1, 3).expand(B, N, 3).contiguous()
        point_cloud = Pointclouds(points=points, features=features)

        # extend to 4 views per object
        point_cloud_expanded = point_cloud.extend(4)  # (B*4)

        cameras = self._mvdream_to_pytorch3d_cameras(camera, dist_scale=dist_scale, view_order="frbl")

        # renderer output: (B*4, H, W, 3) in [0, 1]
        rendered_rgb = self.renderer(point_cloud_expanded, cameras=cameras)

        out = rendered_rgb.permute(0, 3, 1, 2) * 2.0 - 1.0

        return out


class MeshRendererMVDream(MVDreamRenderer):
    def __init__(self, device='cuda', image_size=256, fov_deg=50.0):
        super().__init__(device, image_size, fov_deg)
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

    @torch.no_grad()
    def render_mvdream_views(self, verts, faces, camera, dist_scale=2.5):
        if verts.ndim == 2:
            verts = verts.unsqueeze(0)   # (1, N, 3)
        if faces.ndim == 2:
            faces = faces.unsqueeze(0)   # (1, F, 3)

        B, Nverts, _ = verts.shape
        _, Nfaces, _ = faces.shape
        # Normalize vertices: center and scale to unit sphere
        center = verts.mean(dim=1, keepdim=True)  # (B, 1, 3)
        verts = verts - center
        max_dist = torch.sqrt((verts ** 2).sum(dim=2)).max(dim=1, keepdim=True)[0]  # (B, 1)
        verts = verts / max_dist.unsqueeze(2)

        # sanity
        assert faces.max() < Nverts, (faces.max().item(), Nverts)

        pink = torch.tensor([1.0, 0.2, 0.8], device=self.device)
        verts_rgb = pink.view(1, 1, 3).expand(B, Nverts, 3).contiguous()
        textures = TexturesVertex(verts_features=verts_rgb)

        mesh = Meshes(verts=verts, faces=faces, textures=textures)

        cameras = self._mvdream_to_pytorch3d_cameras(camera, dist_scale=dist_scale, view_order="frbl")
        Vviews = cameras.R.shape[0]

        mesh_expanded = mesh.extend(Vviews)

        rendered = self.renderer(mesh_expanded, cameras=cameras)
        alpha = rendered[..., 3]
        print("alpha max:", alpha.max().item())  # keep until you see >0

        rgb = rendered[..., :3]
        out = rgb.permute(0, 3, 1, 2) * 2.0 - 1.0
        return out