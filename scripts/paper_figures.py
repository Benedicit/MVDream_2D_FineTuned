"""Publication-quality point cloud figures for the paper.

Two figure types:
  1. `make_input_multiview_figure`  - one partial input point cloud rendered from
     several viewpoints, with a column header per view.
  2. `make_comparison_grid_figure`  - a grid of randomly sampled objects with
     columns Input | <method 1> | <method 2> | ... | Ground Truth.

Run directly with the project's conda env, e.g.:
    /home/stud/weisb/miniconda3/envs/ShapeDream/bin/python paper_figures.py --mode both --num_objects 6 --seed 0
"""

import os
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("PYTORCH3D_IGNORE_BIN_SIZE_WARNING", "1")

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer,
    FoVPerspectiveCameras,
    look_at_view_transform,
)
from pytorch3d.io import load_ply, load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes

from view_renderer import normalize_vertices
from util import load_pcd_to_tensor

script_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT_ROOT = f"{script_dir}/../../data/input_pc/chair"
DEFAULT_ADAPOINTR_ROOT = f"{script_dir}/../../data/adapointr_out/chair"
DEFAULT_METHOD_ROOT = f"{script_dir}/debug"
DEFAULT_GT_ROOT = f"{script_dir}/../../data/.pcn/ShapeNetCompletion/val/complete/03001627"
DEFAULT_OUT_DIR = f"{script_dir}/paper_figures_out"

FIXED_VIEW_NAMES = {4: ["Front", "Right", "Back", "Left"]}

# azim=45 in the (GT-aligned) frame gives a clean 3/4-front view for this dataset's
# canonical orientation -- confirmed visually against ground truth.
DEFAULT_FRONTAL_AZIM = 45.0


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


_setup_style()


def _get_cmap(name):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return matplotlib.cm.get_cmap(name)


class FigureRenderer:
    """Renders point clouds with PyTorch3D's point rasterizer + alpha compositor,
    which handles depth ordering correctly (unlike matplotlib's 3D scatter)."""

    def __init__(self, device="cuda", image_size=512, radius=0.008):
        self.device = device
        self.image_size = image_size
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=radius,
            points_per_pixel=8,
        )
        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=(1.0, 1.0, 1.0)),
        ).to(device)

    @torch.no_grad()
    def render_views(self, points, elevs, azims, dist=2.5, fov_deg=50.0, cmap_name="viridis"):
        """points: (N,3) tensor. elevs/azims: sequences of equal length V (degrees).
        Returns (V,H,W,3) uint8 numpy array."""
        assert len(elevs) == len(azims)
        num_views = len(elevs)

        points = points.to(self.device).float().unsqueeze(0)  # (1,N,3)
        points = normalize_vertices(points)

        z = points[0, :, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        cmap = _get_cmap(cmap_name)
        colors = torch.from_numpy(cmap(z_norm.cpu().numpy())[:, :3]).float().to(self.device)
        colors = colors.unsqueeze(0)  # (1,N,3)

        # extend() duplicates the single cloud so each of the `num_views` cameras
        # below renders the same geometry from a different angle.
        point_cloud = Pointclouds(points=points, features=colors).extend(num_views)

        R, T = look_at_view_transform(dist=dist, elev=list(elevs), azim=list(azims))
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov_deg, znear=0.01, zfar=10.0)

        rendered = self.renderer(point_cloud, cameras=cameras)
        rgb = rendered[..., :3].clamp(0, 1)
        return (rgb.cpu().numpy() * 255).astype(np.uint8)

    def render_single_view(self, points, elev, azim, **kwargs):
        return self.render_views(points, [elev], [azim], **kwargs)[0]


# Ground truth (data/.pcn/.../*.pcd) uses PCN's canonical ShapeNet frame. The
# partial input, AdaPoinTr output, and our mesh output instead inherit the raw
# ShapeNetCore mesh export frame, which sits at a fixed -90 deg rotation about
# the up axis relative to the GT frame (verified empirically via nearest-neighbor
# alignment search against GT across multiple objects). Correcting it here means
# a single camera pose renders all four columns consistently everywhere they're used.
_NON_GT_FRAME_OFFSET_DEG = 270.0


def _rotate_up_axis(points, degrees):
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=points.dtype)
    return points @ R.T


def load_input_pc(obj_id, root=DEFAULT_INPUT_ROOT):
    points, _ = load_ply(str(Path(root) / f"{obj_id}.ply"))
    return _rotate_up_axis(points.float(), _NON_GT_FRAME_OFFSET_DEG)


def load_adapointr_pc(obj_id, root=DEFAULT_ADAPOINTR_ROOT):
    points = np.load(str(Path(root) / obj_id / "fine.npy"))
    return _rotate_up_axis(torch.from_numpy(points).float(), _NON_GT_FRAME_OFFSET_DEG)


def load_method_pc(obj_id, root=DEFAULT_METHOD_ROOT, num_samples=16384):
    mesh = load_objs_as_meshes([str(Path(root) / obj_id / "mesh.obj")], device="cpu")
    points = sample_points_from_meshes(mesh, num_samples=num_samples)[0]
    return _rotate_up_axis(points.float(), _NON_GT_FRAME_OFFSET_DEG)


def load_gt_pc(obj_id, root=DEFAULT_GT_ROOT):
    root = Path(root)
    if (root / f"{obj_id}.pcd").exists():
        return load_pcd_to_tensor(root / f"{obj_id}.pcd").float()
    if (root / f"{obj_id}.ply").exists():
        points, _ = load_ply(str(root / f"{obj_id}.ply"))
        return points.float()
    raise FileNotFoundError(f"No ground-truth point cloud for '{obj_id}' in {root}")


DEFAULT_METHODS = [
    {"name": "AdaPoinTr", "root": DEFAULT_ADAPOINTR_ROOT, "output_file": "fine.npy", "loader": load_adapointr_pc},
    {"name": "Ours", "root": DEFAULT_METHOD_ROOT, "output_file": "mesh.obj", "loader": load_method_pc},
]


def discover_common_ids(input_root, method_configs, gt_root):
    """Object IDs present in input_root, every method_configs root, and gt_root."""
    ids = {p.stem for p in Path(input_root).glob("*.ply")}

    for cfg in method_configs:
        root = Path(cfg["root"])
        method_ids = {d.name for d in root.iterdir() if d.is_dir() and (d / cfg["output_file"]).exists()}
        ids &= method_ids

    gt_root = Path(gt_root)
    gt_ids = {p.stem for p in gt_root.glob("*.pcd")} | {p.stem for p in gt_root.glob("*.ply")}
    ids &= gt_ids

    return sorted(ids)


def sample_object_ids(ids, num_objects, seed=None):
    if num_objects > len(ids):
        raise ValueError(f"Requested {num_objects} objects but only {len(ids)} have data in every configured folder.")
    return random.Random(seed).sample(ids, num_objects)


def make_input_multiview_figure(
    obj_id=None,
    input_root=DEFAULT_INPUT_ROOT,
    num_views=4,
    elev=20.0,
    dist=2.5,
    azim_start=DEFAULT_FRONTAL_AZIM,
    out_path=None,
    seed=None,
    renderer=None,
    image_size=512,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    renderer = renderer or FigureRenderer(device=device, image_size=image_size)

    if obj_id is None:
        candidates = sorted(p.stem for p in Path(input_root).glob("*.ply"))
        if not candidates:
            raise FileNotFoundError(f"No .ply files found in {input_root}")
        obj_id = random.Random(seed).choice(candidates)

    points = load_input_pc(obj_id, input_root)
    azims = (np.linspace(0, 360, num_views, endpoint=False) + azim_start) % 360
    views = renderer.render_views(points, elevs=[elev] * num_views, azims=list(azims), dist=dist)
    col_names = FIXED_VIEW_NAMES.get(num_views, [f"View {i + 1} ({a:.0f}°)" for i, a in enumerate(azims)])

    fig, axes = plt.subplots(1, num_views, figsize=(3 * num_views, 3.3))
    if num_views == 1:
        axes = [axes]
    for ax, img, name in zip(axes, views, col_names):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(name)
    fig.tight_layout()

    out_path = out_path or f"{DEFAULT_OUT_DIR}/input_multiview"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{out_path}.png", f"{out_path}.pdf"


def make_comparison_grid_figure(
    num_objects=6,
    object_ids=None,
    input_root=DEFAULT_INPUT_ROOT,
    method_configs=None,
    gt_root=DEFAULT_GT_ROOT,
    elev=20.0,
    azim=DEFAULT_FRONTAL_AZIM,
    dist=2.5,
    out_path=None,
    seed=None,
    renderer=None,
    image_size=512,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    renderer = renderer or FigureRenderer(device=device, image_size=image_size)
    method_configs = method_configs or DEFAULT_METHODS

    if object_ids is None:
        common_ids = discover_common_ids(input_root, method_configs, gt_root)
        if not common_ids:
            raise RuntimeError(
                "No object ID has data in all of input_root, every method_configs root, and gt_root. "
                "Check the configured paths."
            )
        object_ids = sample_object_ids(common_ids, num_objects, seed=seed)
    else:
        num_objects = len(object_ids)

    col_names = ["Input"] + [cfg["name"] for cfg in method_configs] + ["Ground Truth"]
    n_cols = len(col_names)

    fig, axes = plt.subplots(num_objects, n_cols, figsize=(2.6 * n_cols, 2.6 * num_objects))
    if num_objects == 1:
        axes = axes[None, :]

    for row, obj_id in enumerate(object_ids):
        cell_points = [load_input_pc(obj_id, input_root)]
        cell_points += [cfg["loader"](obj_id, cfg["root"]) for cfg in method_configs]
        cell_points += [load_gt_pc(obj_id, gt_root)]

        for col, points in enumerate(cell_points):
            img = renderer.render_single_view(points, elev=elev, azim=azim, dist=dist)
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_names[col])

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    out_path = out_path or f"{DEFAULT_OUT_DIR}/comparison_grid"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{out_path}.png", f"{out_path}.pdf"


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-quality point cloud figures.")
    parser.add_argument("--mode", choices=["input_views", "comparison", "both"], default="both")
    parser.add_argument("--num_objects", type=int, default=6)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=DEFAULT_FRONTAL_AZIM)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--adapointr_root", type=str, default=DEFAULT_ADAPOINTR_ROOT)
    parser.add_argument("--method_root", type=str, default=DEFAULT_METHOD_ROOT)
    parser.add_argument("--gt_root", type=str, default=DEFAULT_GT_ROOT)
    parser.add_argument("--object_id", type=str, default=None, help="Force a specific object id for input_views mode.")
    return parser.parse_args()


def main():
    args = _parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    renderer = FigureRenderer(device=device, image_size=args.image_size)

    method_configs = [
        {"name": "AdaPoinTr", "root": args.adapointr_root, "output_file": "fine.npy", "loader": load_adapointr_pc},
        {"name": "Ours", "root": args.method_root, "output_file": "mesh.obj", "loader": load_method_pc},
    ]

    if args.mode in ("input_views", "both"):
        png, pdf = make_input_multiview_figure(
            obj_id=args.object_id,
            input_root=args.input_root,
            num_views=args.num_views,
            elev=args.elev,
            azim_start=args.azim,
            out_path=f"{args.out_dir}/input_multiview",
            seed=args.seed,
            renderer=renderer,
        )
        print(f"Saved input multiview figure to {png} / {pdf}")

    if args.mode in ("comparison", "both"):
        png, pdf = make_comparison_grid_figure(
            num_objects=args.num_objects,
            input_root=args.input_root,
            method_configs=method_configs,
            gt_root=args.gt_root,
            elev=args.elev,
            azim=args.azim,
            out_path=f"{args.out_dir}/comparison_grid",
            seed=args.seed,
            renderer=renderer,
        )
        print(f"Saved comparison grid figure to {png} / {pdf}")


if __name__ == "__main__":
    main()
