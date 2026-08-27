from __future__ import annotations

import os
from typing import Mapping

import numpy as np
import torch

from pytorch3d.ops import sample_farthest_points
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.io import load_objs_as_meshes, load_ply, save_ply
from tqdm import tqdm
__all__ = ["generate_partial_shapes"]


def _random_views(n, dist_range, generator, device):
    """`n` poses looking at the origin (PCN's random_pose): viewpoint uniform on the sphere,
    random up-vector so all camera rolls occur."""
    u = torch.rand(n, generator=generator) * 2.0 - 1.0
    elev = torch.rad2deg(torch.asin(u))
    azim = torch.rand(n, generator=generator) * 360.0 - 180.0
    dist = dist_range[0] + torch.rand(n, generator=generator) * (dist_range[1] - dist_range[0])

    er, ar = torch.deg2rad(elev), torch.deg2rad(azim)
    eye = torch.stack([er.cos() * ar.sin(), er.sin(), er.cos() * ar.cos()], dim=-1)

    up = torch.randn(n, 3, generator=generator)
    up = up / up.norm(dim=1, keepdim=True)
    for _ in range(32):  # redraw up-vectors nearly parallel to the view axis
        bad = (up * eye).sum(-1).abs() > 0.95
        if not bool(bad.any()):
            break
        repl = torch.randn(int(bad.sum()), 3, generator=generator)
        up[bad] = repl / repl.norm(dim=1, keepdim=True)

    return look_at_view_transform(dist=dist, elev=elev, azim=azim, up=up, device=device)


def _render_batch(clouds, cameras, rasterizer, x_ndc, y_ndc, mode):
    """One depth render per (cloud, camera) pair -> list of world-space partial clouds.
    mode="depth" back-projects the z-buffer (PCN's process_exr.py), mode="visible" keeps the
    input points that pass the z-test."""
    pcl = Pointclouds(points=clouds)
    fragments = rasterizer(pcl, cameras=cameras)
    zbuf = fragments.zbuf[..., 0]  # (N, H, W), -1 where no splat covers the pixel
    valid = zbuf > 0
    n = zbuf.shape[0]

    if mode == "visible":
        idx = fragments.idx[..., 0].long()  # indexes the packed points of the batch
        first = pcl.cloud_to_packed_first_idx()
        return [clouds[i][torch.unique(idx[i][valid[i]] - first[i])] for i in range(n)]

    # pixel NDC + depth -> camera coords -> world coords; invalid pixels masked out after
    xy = torch.stack([x_ndc, y_ndc], dim=-1).reshape(1, -1, 2).expand(n, -1, -1)
    z = torch.where(valid, zbuf, torch.ones_like(zbuf)).reshape(n, -1, 1)
    world = cameras.unproject_points(
        torch.cat([xy, z], dim=-1), world_coordinates=True, scaled_depth_input=False
    )
    flat = valid.reshape(n, -1)
    return [world[i][flat[i]] for i in range(n)]


@torch.no_grad()
def generate_partial_shapes(
        shapes: Mapping[str, torch.Tensor],
        base_path: str,
        n_points: int = 2048,
        image_size: int = 320,
        fov: float = 60.0,
        dist_range: tuple[float, float] = (2.2, 3.0),
        point_radius_px: float = 2.5,
        mode: str = "depth",
        batch_size: int = 8,
        min_valid_pixels: int = 128,
        max_view_attempts: int = 8,
        use_fps: bool = False,
        device: str | torch.device | None = None,
        seed: int | None = None,
) -> dict[str, str]:
    """
    Write one partial cloud per complete cloud to `base_path/<obj_id>.pc`, and return
    {obj_id: path}.

    shapes:            {obj_id: (N, 3) tensor}; obj_id may contain "/".
    n_points:          points per partial cloud.
    image_size:        square depth-image resolution.
    fov:               vertical field of view in degrees.
    dist_range:        camera distance in units of the object's bounding-sphere radius; the
                       default keeps the object inside a 60 deg frame (asin(1/2.2) = 27 deg).
    point_radius_px:   splat radius; too small leaks back-facing points, too large bloats
                       silhouettes.
    mode:              "depth" (PCN-style back-projection) or "visible" (z-test filter).
    min_valid_pixels:  redraw the viewpoint below this many valid pixels.
    use_fps:           farthest-point sampling instead of PCN's uniform resampling.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    generator = torch.Generator()  # CPU generator -> reproducible on any device
    if seed is not None:
        generator.manual_seed(seed)
    os.makedirs(base_path, exist_ok=True)

    # pixel centres in NDC, PyTorch3D convention: +X left, +Y up
    ndc = 1.0 - (2.0 * torch.arange(image_size, device=device, dtype=torch.float32) + 1.0) / image_size
    y_ndc, x_ndc = torch.meshgrid(ndc, ndc, indexing="ij")

    rasterizer = PointsRasterizer(
        cameras=FoVPerspectiveCameras(device=device),  # placeholder, overridden per call
        raster_settings=PointsRasterizationSettings(
            image_size=image_size,
            radius=point_radius_px * 2.0 / image_size,  # pixels -> NDC
            points_per_pixel=1,
            bin_size=0,
        ),
    )
    cam_kwargs = dict(fov=fov, znear=0.1, zfar=100.0, device=device)

    items = list(shapes.items())
    written: dict[str, str] = {}

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        norm, centroids, scales = [], [], []
        for _, cloud in chunk:
            pts = cloud.to(device=device, dtype=torch.float32)
            centroid = pts.mean(0)
            scale = (pts - centroid).norm(dim=1).max().clamp_min(1e-8)  # into the unit sphere
            norm.append((pts - centroid) / scale)
            centroids.append(centroid)
            scales.append(scale)

        R, T = _random_views(len(chunk), dist_range, generator, device)
        cameras = FoVPerspectiveCameras(R=R, T=T, **cam_kwargs)
        partials = _render_batch(norm, cameras, rasterizer, x_ndc, y_ndc, mode)
        progressbar = tqdm(total=len(chunk), desc="Generating partials", unit="obj")
        for k, (obj_id, _) in enumerate(chunk):
            partial = partials[k]
            for _ in range(max_view_attempts - 1):  # degenerate view, e.g. a plane edge-on
                if partial.shape[0] >= min_valid_pixels:
                    break
                R1, T1 = _random_views(1, dist_range, generator, device)
                retry = _render_batch(
                    [norm[k]], FoVPerspectiveCameras(R=R1, T=T1, **cam_kwargs),
                    rasterizer, x_ndc, y_ndc, mode,
                )[0]
                partial = retry if retry.shape[0] > partial.shape[0] else partial

            partial = partial * scales[k] + centroids[k]  # back to the input frame

            p = partial.shape[0]  # PCN's resample_pcd: drop or duplicate to exactly n_points
            if use_fps and p >= n_points:
                partial = sample_farthest_points(partial[None], K=n_points, random_start_point=True)[0][0]
            else:
                idx = torch.randperm(p, generator=generator)
                if p < n_points:
                    idx = torch.cat([idx, torch.randint(0, p, (n_points - p,), generator=generator)])
                partial = partial[idx[:n_points].to(device)]

            path = os.path.join(base_path, f"{obj_id}.ply")
            save_ply(path, partial)
            written[obj_id] = path
            progressbar.update(1)

    return written