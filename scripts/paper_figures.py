"""Publication-quality point cloud figures for the paper.

Two figure types:
  1. `make_input_multiview_figure`  - one partial input point cloud rendered from
     several viewpoints, with a column header per view.
  2. `make_comparison_grid_figure`  - a grid of objects, sampled at random or, with
     --select, the ones our model reconstructs most faithfully, with columns
     Input | <baselines> | <our models> | Ground Truth.

Two evaluation universes, selected with --universe. Both cover the PCN *test* split
objects; they differ in which partial scan the methods consumed.

  pcn_test (PCN's own partials)
    Input       .pcn/ShapeNetCompletion/test/partial/<taxonomy>/<id>/00.pcd
    AdaPoinTr   data/adapointr_out/pcn_test/<class_name>/<id>/fine.npy
    SVDFormer   data/svdformer_out/pcn_test/<class_name>/<id>/fine.npy
    Ours        <rank>_lora/<taxonomy>/<id>/mesh.obj
    GT          .pcn/ShapeNetCompletion/test/complete/<taxonomy>/<id>.pcd

  custom_input (our generated scans, see data/baseline_data_gen.py)
    Input       data/input_pc/<class_name>/<id>.ply
    AdaPoinTr   data/adapointr_out/custom_input/<class_name>/<id>/fine.npy
    SVDFormer   data/svdformer_out/custom_input/<class_name>/<id>/fine.npy
    Ours        <rank>_lora/custom_input/<class_name>/<id>/mesh.obj
    GT          same as above

Two things vary between them, and custom_input is uniform on both:
  - Folder key. In custom_input every column is keyed by class NAME except GT, which
    stays on taxonomy id. pcn_test is mixed: baselines by class name, the rest by
    taxonomy id.
  - Frame. The raw ShapeNetCore frame is 270 deg about the up axis from the PCN/GT
    frame. In pcn_test only our meshes sit there; in custom_input the scans and
    everything that consumed them do, so every column but GT is rotated.

Any method whose directory is missing is silently dropped from the figure - a
LoRA rank that has not been run yet, or a baseline with no data for a class,
simply produces no column. Classes that lack input/GT/our-model data are dropped
entirely rather than rendered as blank panels.

--select scores candidates the way data/loss.py does - UHD against the partial input,
F-Score against the ground truth - and keeps the best. custom_input is scored against
the ShapeNet mesh the scans were cut from, not against PCN's cloud, which sits a bbox
centre away from it. Scores land in <out_dir>/fidelity_scores.csv; delete that file
after re-running inference.

Run directly with the project's conda env, e.g.:
    /home/stud/weisb/miniconda3/envs/ShapeDream/bin/python paper_figures.py \
        --mode comparison --universe custom_input --classes all --num_per_class 2 --seed 0
"""

import os
import argparse
import csv
import math
import random
import warnings
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Collection, Sequence

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("PYTORCH3D_IGNORE_BIN_SIZE_WARNING", "1")

from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer,
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    HardPhongShader,
    AmbientLights,
    TexturesVertex,
    BlendParams,
)
from pytorch3d.io import load_ply, load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points, iterative_closest_point

from view_renderer import normalize_vertices
from util import load_pcd_to_tensor

# Every mesh.obj is exported without a material file; the warning is harmless but
# would otherwise fire once per rendered cell.
warnings.filterwarnings("ignore", message="No mtl file provided")

script_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(script_dir).parents[1]

DEFAULT_PCN_ROOT = REPO_ROOT / "data/.pcn/ShapeNetCompletion"
DEFAULT_SPLIT = "test"
DEFAULT_UNIVERSE = "pcn_test"
#DEFAULT_UNIVERSE = "custom_input"
UNIVERSES = ("pcn_test", "custom_input")
DEFAULT_INPUT_PC_ROOT = REPO_ROOT / "data/input_pc"
DEFAULT_SHAPENET_ROOT = REPO_ROOT / "data/.shapenet"
DEFAULT_LABEL_CSV = REPO_ROOT / "data/shapenet_label_to_mesh.csv"
DEFAULT_ADAPOINTR_ROOT = REPO_ROOT / "data/adapointr_out"
DEFAULT_SVDFORMER_ROOT = REPO_ROOT / "data/svdformer_out"
DEFAULT_LORA_ROOT = Path(script_dir)
DEFAULT_LORA_RANKS = (32, 64, 80)
DEFAULT_OUT_DIR = f"{script_dir}/paper_figures_out"

FIXED_VIEW_NAMES = {4: ["Front", "Right", "Back", "Left"]}

# All columns are normalised into the PCN/GT frame before rendering, so the
# azimuth is interpreted in that frame. 45 deg gives a clean 3/4-front view.
DEFAULT_FRONTAL_AZIM = 45.0

# The raw ShapeNetCore frame sits at a fixed -90 deg about the up axis relative to
# the PCN/GT frame (verified by nearest-neighbour alignment search against GT across
# several classes). In pcn_test only our LoRA meshes need it; in custom_input the
# scans and everything that consumed them are in that frame too, so every column but
# the ground truth is rotated. Rotation about +Y commutes with normalize_vertices, so
# applying it at load time is equivalent to applying it after normalisation.
SHAPENET_FRAME_OFFSET_DEG = 270.0


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        #"font.size": 11,
        "font.size": 20,
        #"axes.titlesize": 12,
        "axes.titlesize": 20,
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


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassSpec:
    taxonomy_id: str
    name: str
    is_pcn: bool = True


# The 8 PCN classes. Hardcoded rather than parsed from AdaPoinTr's PCN.json: the
# only unique content there is this id<->name map (the object ids come from
# globbing the method roots), PCN is a frozen benchmark, and a figure script for
# this repo should not depend on a sibling checkout's data directory.
PCN_CLASSES = (
    ClassSpec("02691156", "airplane"),
    ClassSpec("02933112", "cabinet"),
    ClassSpec("02958343", "car"),
    ClassSpec("03001627", "chair"),
    ClassSpec("03636649", "lamp"),
    ClassSpec("04256520", "sofa"),
    ClassSpec("04379243", "table"),
    ClassSpec("04530566", "watercraft"),
)

# Classes outside PCN's 8, run through the same custom_input pipeline. Their ground
# truth is the source ShapeNet mesh, not a PCN complete cloud -- see _GtMethod.
OOD_CLASSES = (
    ClassSpec("02808440", "bathtub", is_pcn=False),
    ClassSpec("02818832", "bed", is_pcn=False),
    ClassSpec("02876657", "bottle", is_pcn=False),
    ClassSpec("03593526", "jar", is_pcn=False),
    ClassSpec("03991062", "flowerpot", is_pcn=False),
    ClassSpec("03642806", "laptop", is_pcn=False),
)

_ALL_CLASSES = PCN_CLASSES #+ OOD_CLASSES
_BY_NAME = {c.name: c for c in _ALL_CLASSES}
_BY_TAX = {c.taxonomy_id: c for c in _ALL_CLASSES}


def resolve_classes(tokens: Sequence[str]) -> list[ClassSpec]:
    """Map CLI tokens (class names, taxonomy ids, 'all' or 'ood') to ClassSpecs.

    Unknown 8-digit tokens are accepted as non-PCN classes; anything else is a
    hard error. Silent skipping applies to missing *data*, never to user input.
    """
    if not tokens or "all" in tokens:
        return list(PCN_CLASSES)
    if "ood" in tokens:
        return list(OOD_CLASSES)

    out, seen = [], set()
    for tok in tokens:
        if tok in _BY_NAME:
            cls = _BY_NAME[tok]
        elif tok in _BY_TAX:
            cls = _BY_TAX[tok]
        elif tok.isdigit() and len(tok) == 8:
            cls = ClassSpec(tok, tok, is_pcn=False)
        else:
            raise SystemExit(
                f"Unknown class '{tok}'. Valid names: {', '.join(sorted(_BY_NAME))}; "
                f"or an 8-digit taxonomy id, or 'all'/'ood'."
            )
        if cls.taxonomy_id not in seen:
            seen.add(cls.taxonomy_id)
            out.append(cls)

    # Keep canonical order so figures are stable regardless of CLI ordering.
    order = {c.taxonomy_id: i for i, c in enumerate(_ALL_CLASSES)}
    return sorted(out, key=lambda c: (order.get(c.taxonomy_id, len(order)), c.taxonomy_id))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class MeshPayload:
    verts: torch.Tensor  # (V,3)
    faces: torch.Tensor  # (F,3) long


@dataclass
class LoadOpts:
    render_ours_as_mesh: bool = False
    mesh_samples: int = 16384


def _rotate_up_axis(points, degrees):
    if not degrees:
        return points
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=points.dtype)
    return points @ R.T


def load_pcd_points(path, opts):
    return load_pcd_to_tensor(path).float()


def load_npy_points(path, opts):
    return torch.from_numpy(np.load(str(path))).float()


def load_ply_points(path, opts):
    points, _ = load_ply(str(path))
    return points.float()


def load_obj_payload(path, opts):
    """Our models emit a mesh; either sample it into a point cloud (default, so
    every column shares one visual language) or keep it for surface rendering."""
    mesh = load_objs_as_meshes([str(path)], device="cpu")
    if opts.render_ours_as_mesh:
        return MeshPayload(verts=mesh.verts_packed().float(), faces=mesh.faces_packed().long())
    return sample_points_from_meshes(mesh, num_samples=opts.mesh_samples)[0].float()


_OOD_MESHES = {}


def _ood_mesh_index(cls) -> dict:
    """{obj_id: model_normalized.obj} for a non-PCN class.

    PCN-class object ids are ShapeNet hashes and index the mesh tree directly;
    OOD ids are <class><n> and only the label CSV knows which mesh they came from.
    """
    if cls.taxonomy_id not in _OOD_MESHES:
        found = {}
        if DEFAULT_LABEL_CSV.is_file():
            with open(DEFAULT_LABEL_CSV, newline="") as f:
                for row in csv.DictReader(f):
                    if row["label"] == cls.name:
                        found[row["pc_id"].removeprefix("shapenet_")] = Path(row["filename"])
        _OOD_MESHES[cls.taxonomy_id] = found
    return _OOD_MESHES[cls.taxonomy_id]


def mesh_path(shapenet_root, cls, obj_id):
    if not cls.is_pcn:
        return _ood_mesh_index(cls).get(obj_id)
    return Path(shapenet_root) / cls.taxonomy_id / obj_id / "models/model_normalized.obj"


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

@dataclass
class Method:
    key: str
    display_name: str
    root: Path
    kind: str                    # "input" | "baseline" | "ours" | "gt"
    folder_key: str              # "taxonomy_id" | "class_name"
    artifact_glob: str           # relative to the class dir
    id_of: Callable[[Path], str]
    loader: Callable[[Path, LoadOpts], object]
    rotate_deg: float = 0.0
    _index: dict = field(default_factory=dict, repr=False)

    def class_dir(self, cls: ClassSpec) -> Path:
        key = cls.taxonomy_id if self.folder_key == "taxonomy_id" else cls.name
        return self.root / key

    def has_class(self, cls: ClassSpec) -> bool:
        return self.class_dir(cls).is_dir()

    def index(self, cls: ClassSpec) -> dict:
        """{obj_id: path} for this class, globbed once and cached.

        Discovery and loading share this dict by construction, so "id is in the
        index" always means "the file existed when we snapshotted".
        """
        if cls.taxonomy_id in self._index:
            return self._index[cls.taxonomy_id]
        found = {}
        if self.has_class(cls):
            for p in self.class_dir(cls).glob(self.artifact_glob):
                found[self.id_of(p)] = p
        self._index[cls.taxonomy_id] = found
        return found

    def load(self, cls: ClassSpec, obj_id: str, opts: LoadOpts):
        payload = self.loader(self.index(cls)[obj_id], opts)
        if isinstance(payload, MeshPayload):
            return replace(payload, verts=_rotate_up_axis(payload.verts, self.rotate_deg))
        return _rotate_up_axis(payload, self.rotate_deg)


@dataclass
class _GtMethod(Method):
    """PCN complete clouds, falling back to the source ShapeNet mesh for OOD classes,
    which PCN has no ground truth for. Stays a point cloud either way."""

    def has_class(self, cls):
        return super().has_class(cls) if cls.is_pcn else bool(_ood_mesh_index(cls))

    def index(self, cls):
        return super().index(cls) if cls.is_pcn else _ood_mesh_index(cls)

    def load(self, cls, obj_id, opts):
        if cls.is_pcn:
            return super().load(cls, obj_id, opts)
        points = load_obj_payload(self.index(cls)[obj_id],
                                  replace(opts, render_ours_as_mesh=False))
        return _rotate_up_axis(points, SHAPENET_FRAME_OFFSET_DEG)


def build_registry(
    *,
    universe=DEFAULT_UNIVERSE,
    pcn_root=DEFAULT_PCN_ROOT,
    split=DEFAULT_SPLIT,
    input_pc_root=DEFAULT_INPUT_PC_ROOT,
    adapointr_root=DEFAULT_ADAPOINTR_ROOT,
    svdformer_root=DEFAULT_SVDFORMER_ROOT,
    lora_root=DEFAULT_LORA_ROOT,
    lora_ranks=DEFAULT_LORA_RANKS,
    exclude: Collection[str] = (),
) -> list[Method]:
    """Methods in fixed column order: Input | baselines | ours | Ground Truth.

    Two universes, differing only in where the input comes from and how the
    per-class folders are keyed:

      pcn_test      PCN's own partials; baselines keyed by class name, input/ours/GT
                    by taxonomy id; only our meshes are in the ShapeNetCore frame.
      custom_input  our generated scans; every column keyed by class name except GT,
                    and every column in the ShapeNetCore frame except GT.

    A method whose root directory does not exist is omitted entirely, which is
    what makes an un-run LoRA rank or a missing baseline silently disappear.
    """
    pcn_root, lora_root = Path(pcn_root), Path(lora_root)
    parent_name = lambda p: p.parent.name

    if universe == "custom_input":
        rot = SHAPENET_FRAME_OFFSET_DEG
        input_method = Method("input", "Input", Path(input_pc_root), "input",
                              "class_name", "*.ply", lambda p: p.stem, load_ply_points,
                              rotate_deg=rot)
        lora_sub, lora_key = "custom_input", "class_name"
    else:
        rot = 0.0
        input_method = Method("input", "Input", pcn_root / split / "partial", "input",
                              "taxonomy_id", "*/00.pcd", parent_name, load_pcd_points)
        lora_sub, lora_key = "", "taxonomy_id"

    candidates = [
        input_method,
        Method("adapointr", "AdaPoinTr", Path(adapointr_root) / universe, "baseline",
               "class_name", "*/fine.npy", parent_name, load_npy_points, rotate_deg=rot),
        Method("svdformer", "SVDFormer", Path(svdformer_root) / universe, "baseline",
               "class_name", "*/fine.npy", parent_name, load_npy_points, rotate_deg=rot),
    ]
    for rank in sorted(lora_ranks):
        candidates.append(
            Method(f"lora{rank}", f"Ours (r={rank})", lora_root / f"{rank}_lora" / lora_sub,
                   "ours", lora_key, "*/mesh.obj", parent_name, load_obj_payload,
                   rotate_deg=SHAPENET_FRAME_OFFSET_DEG)
        )
    candidates.append(
        _GtMethod("gt", "GT", pcn_root / split / "complete", "gt",
                  "taxonomy_id", "*.pcd", lambda p: p.stem, load_pcd_points)
    )

    return [m for m in candidates if m.key not in exclude and m.root.is_dir()]


# ---------------------------------------------------------------------------
# Grid resolution
# ---------------------------------------------------------------------------

@dataclass
class GridPlan:
    columns: list
    rows: list                  # [(ClassSpec, obj_id), ...] grouped by class
    fallbacks: dict = field(default_factory=dict)   # taxonomy_id -> spare ids
    notes: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)      # (taxonomy_id, obj_id) -> (UHD, F-Score)


def _class_ids(columns, cls):
    """Object ids present in every column that serves this class."""
    sets = [set(m.index(cls)) for m in columns if m.has_class(cls)]
    return sorted(set.intersection(*sets)) if sets else []


def resolve_grid(methods, classes, num_per_class, seed=0, pinned=None, ranker=None) -> GridPlan:
    """Pick columns and rows, dropping whatever has no data.

    Columns are intersected across the surviving classes rather than unioned:
    a blank cell in a method column reads as "this method produced nothing",
    which would be an unearned win. An absent column is honest, a hole is not.
    """
    classes = list(classes)
    notes = []

    def mandatory_ok(c):
        kinds = {m.kind for m in methods if m.has_class(c)}
        return {"input", "gt"}.issubset(kinds) and "ours" in kinds

    while True:
        keep = [c for c in classes if mandatory_ok(c)]
        for c in classes:
            if c not in keep:
                notes.append(f"{c.name}: dropped (missing input, ground truth, or any of our models)")
        classes = keep
        if not classes:
            break

        # A baseline must serve every surviving class or it is dropped outright.
        columns = [m for m in methods
                   if m.kind != "baseline" or all(m.has_class(c) for c in classes)]
        columns = [m for m in columns if any(m.has_class(c) for c in classes)]

        dropped = False
        for c in list(classes):
            if not _class_ids(columns, c):
                notes.append(f"{c.name}: dropped (no object has data in every column)")
                classes.remove(c)
                dropped = True
        if not dropped:
            break

    if not classes:
        raise RuntimeError(
            "No class has data in every required column. Check the configured roots "
            "(--pcn_root / --lora_root / --adapointr_root / --svdformer_root)."
        )

    shown = {m.key for m in columns}
    for m in methods:
        if m.key not in shown:
            notes.append(f"{m.display_name}: column omitted (no data for the selected classes)")

    # If only one of our models survives, plain "Ours" reads better than "Ours (r=64)".
    ours = [m for m in columns if m.kind == "ours"]
    if len(ours) == 1:
        ours[0].display_name = "Ours"

    if pinned is not None and ranker is not None:
        notes.append("rows pinned with --object_ids, --select ignored")

    rows, fallbacks = [], {}
    for cls in classes:
        ids = _class_ids(columns, cls)
        if pinned is not None:
            chosen = [i for i in pinned.get(cls.taxonomy_id, []) if i in set(ids)]
            missing = [i for i in pinned.get(cls.taxonomy_id, []) if i not in set(ids)]
            for i in missing:
                notes.append(f"{cls.name}: pinned id {i} has no data in every column, skipped")
            pool = []
        else:
            # Seed per class so that adding a class does not reshuffle the others; the
            # ranker reorders that sample rather than replacing it, so --seed still
            # decides what --rank_pool scores and how the unranked ones trail behind.
            pool = random.Random(f"{seed}:{cls.taxonomy_id}").sample(ids, len(ids))
            if ranker is not None:
                pool = ranker(cls, pool, columns)
            take = min(num_per_class, len(pool))
            if take < num_per_class:
                notes.append(
                    f"{cls.name}: only {take} of {num_per_class} requested objects "
                    f"have data in every column"
                )
            chosen, pool = pool[:take], pool[take:]

        rows += [(cls, i) for i in chosen]
        fallbacks[cls.taxonomy_id] = pool

    if not rows:
        raise RuntimeError("No rows to render after resolving the grid.")

    return GridPlan(columns=columns, rows=rows, fallbacks=fallbacks, notes=notes,
                    scores=ranker.scores if ranker is not None else {})


def describe_plan(plan: GridPlan) -> str:
    lines = ["Columns: " + " | ".join(m.display_name for m in plan.columns)]
    by_class = {}
    for cls, obj_id in plan.rows:
        by_class.setdefault(cls, []).append(obj_id)
    for cls, ids in by_class.items():
        lines.append(f"  {cls.name} ({cls.taxonomy_id}): {len(ids)} object(s)")
        for i in ids:
            score = plan.scores.get((cls.taxonomy_id, i))
            lines.append(f"      {i}" + (f"   UHD {score[0]:.4f}   F-Score {score[1]:.4f}"
                                         if score else ""))
    for note in plan.notes:
        lines.append(f"  note: {note}")
    pins = " ".join(f"{cls.taxonomy_id}:{i}" for cls, i in plan.rows)
    lines.append("Pin these exact rows with:\n  --object_ids " + pins)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fidelity ranking
# ---------------------------------------------------------------------------

# Scoring follows data/loss.py, so the numbers land on data/metric_results_*.csv:
# sample our mesh, match it to the ground truth's centre and RMS radius, refine with
# ICP, then measure UHD against the partial input and F-Score against the ground
# truth. knn_points returns squared distances, so the threshold is F-Score@1% of the
# unit bbox diagonal.
RANK_POINTS = 16384
RANK_THRESHOLD = 1e-4
RANK_OPTS = LoadOpts(mesh_samples=RANK_POINTS)


def _align_to(points, target):
    """data/loss.py::align_frame, then ICP. The rescale is load-bearing: our
    marching-cubes hull is far larger than the ground truth, and ICP is rigid only."""
    def frame(x):
        lo, hi = x.min(dim=1, keepdim=True)[0], x.max(dim=1, keepdim=True)[0]
        r = (x - x.mean(dim=1, keepdim=True)).norm(dim=2)
        return (lo + hi) / 2, r.pow(2).mean(dim=1).sqrt()[:, None, None]

    centre, size = frame(points)
    t_centre, t_size = frame(target)
    return iterative_closest_point((points - centre) / size * t_size + t_centre, target).Xt


def _fidelity(truth, pred, partial):
    """(UHD, F-Score) for one object, all three clouds in one frame."""
    d_p2g = knn_points(pred, truth, K=1).dists[..., 0]
    d_g2p = knn_points(truth, pred, K=1).dists[..., 0]
    prec = (d_p2g < RANK_THRESHOLD).float().mean(dim=1)
    rec = (d_g2p < RANK_THRESHOLD).float().mean(dim=1)
    uhd = knn_points(partial, pred, K=1).dists[..., 0].max(dim=1)[0].sqrt()
    return uhd.item(), (2 * prec * rec / (prec + rec + 1e-8)).item()


@dataclass
class FidelityRanker:
    """Orders a class's objects by how faithfully our model reconstructs the input.

    Scoring costs about a second per object, so scores are cached on disk. An object
    that fails to score keeps its sampled position behind the ranked ones, which
    costs it a rank but never drops it from the grid.
    """
    universe: str
    ours_key: str
    criterion: str = "best"       # "best" (mean of both ranks) | "uhd" | "fscore"
    pool: int = 0                 # objects scored per class, 0 = all
    shapenet_root: Path = None    # set for custom_input only, see _truth_points
    device: str = "cuda"
    cache: Path = None
    scores: dict = field(default_factory=dict, repr=False)   # (taxonomy_id, id) -> (UHD, F)

    def __post_init__(self):
        self.tag = f"{self.universe}:{self.ours_key}"
        if self.cache and self.cache.is_file():
            with open(self.cache, newline="") as f:
                for r in csv.DictReader(f):
                    if r["tag"] == self.tag:
                        self.scores[(r["taxonomy_id"], r["obj_id"])] = (float(r["uhd"]),
                                                                        float(r["f_score"]))

    def __call__(self, cls, ids, columns):
        ours = next((m for m in columns if m.key == self.ours_key and m.has_class(cls)), None)
        if ours is None:
            print(f"  {cls.name}: {self.ours_key} has no data, keeping the sampled order")
            return ids
        inp = next(m for m in columns if m.kind == "input")
        gt = next(m for m in columns if m.kind == "gt")

        candidates = ids[:self.pool] if self.pool else ids
        todo = [i for i in candidates if (cls.taxonomy_id, i) not in self.scores]
        if todo:
            print(f"  ranking {cls.name}: scoring {len(todo)} objects on {ours.display_name}")
        fresh = []
        for obj_id in todo:
            try:
                score = self._score(cls, obj_id, ours, inp, gt)
            except Exception as exc:
                print(f"    {obj_id}: {type(exc).__name__}: {exc}")
                continue
            self.scores[(cls.taxonomy_id, obj_id)] = score
            fresh.append((self.tag, cls.taxonomy_id, obj_id) + score)
        self._append_cache(fresh)
        torch.cuda.empty_cache()

        ranked = self._order(cls, [i for i in candidates if (cls.taxonomy_id, i) in self.scores])
        return ranked + [i for i in ids if i not in set(ranked)]

    def _order(self, cls, ids):
        """Best first. UHD is a distance and F-Score a ratio, so "best" averages the
        two orderings rather than their values; ties keep the sampled order."""
        def ranks(key):
            return {i: pos for pos, i in enumerate(sorted(ids, key=key))}

        by_uhd = ranks(lambda i: self.scores[(cls.taxonomy_id, i)][0])
        by_f = ranks(lambda i: -self.scores[(cls.taxonomy_id, i)][1])
        rank = {"uhd": lambda i: by_uhd[i],
                "fscore": lambda i: by_f[i],
                "best": lambda i: by_uhd[i] + by_f[i]}[self.criterion]
        return sorted(ids, key=rank)

    @torch.no_grad()
    def _score(self, cls, obj_id, ours, inp, gt):
        # Forked RNG with a per-object seed: sampling here must not shift what the
        # renderer samples later, and a score must not depend on the scoring order.
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(zlib.crc32(obj_id.encode()))
            pred = ours.load(cls, obj_id, RANK_OPTS).to(self.device)[None]
            truth = self._truth_points(cls, obj_id, inp, gt).to(self.device)[None]
        partial = inp.load(cls, obj_id, RANK_OPTS).to(self.device)[None]
        return _fidelity(truth, _align_to(pred, truth), partial)

    def _truth_points(self, cls, obj_id, inp, gt):
        """The ground truth in the input's frame.

        For custom_input that is not the plotted ground truth: the scans were cut from
        the ShapeNet mesh, which PCN's cloud sits a bbox centre away from. ICP would
        absorb that offset on our prediction and leave it in the UHD.
        """
        if self.shapenet_root is None:
            return gt.load(cls, obj_id, RANK_OPTS)
        mesh = mesh_path(self.shapenet_root, cls, obj_id)
        return _rotate_up_axis(load_obj_payload(mesh, RANK_OPTS), inp.rotate_deg)

    def _append_cache(self, rows):
        if not (self.cache and rows):
            return
        header = not self.cache.is_file()
        os.makedirs(self.cache.parent, exist_ok=True)
        with open(self.cache, "a", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(["tag", "taxonomy_id", "obj_id", "uhd", "f_score"])
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class FigureRenderer:
    """Renders point clouds and meshes with PyTorch3D, which handles depth
    ordering correctly (unlike matplotlib's 3D scatter)."""

    def __init__(self, device="cuda", image_size=512, radius=0.008,
                 points_per_pixel=8, max_faces_per_bin=200_000):
        self.device = device
        self.image_size = image_size
        self.max_faces_per_bin = max_faces_per_bin
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=radius,
            points_per_pixel=points_per_pixel,
        )
        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=(1.0, 1.0, 1.0)),
        ).to(device)
        self._mesh_renderers = None

    def _cameras(self, elevs, azims, dist, fov_deg):
        R, T = look_at_view_transform(dist=dist, elev=list(elevs), azim=list(azims))
        return FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov_deg,
                                     znear=0.01, zfar=10.0)

    @staticmethod
    def _height_colors(xyz, cmap_name):
        """Per-vertex/per-point viridis ramp. Shared by the point and mesh paths
        so a mesh column is coloured identically to the point columns."""
        v = xyz[0, :, 2]
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        cmap = _get_cmap(cmap_name)
        return torch.from_numpy(cmap(v.cpu().numpy())[:, :3]).float().to(xyz.device).unsqueeze(0)

    def _build_mesh_renderers(self):
        if self._mesh_renderers is not None:
            return self._mesh_renderers
        lights = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))
        blend = BlendParams(background_color=(1.0, 1.0, 1.0))
        shader = HardPhongShader(device=self.device, lights=lights, blend_params=blend)
        heuristic = RasterizationSettings(image_size=self.image_size, blur_radius=0.0,
                                          faces_per_pixel=1, bin_size=None,
                                          max_faces_per_bin=self.max_faces_per_bin)
        naive = RasterizationSettings(image_size=self.image_size, blur_radius=0.0,
                                      faces_per_pixel=1, bin_size=0)
        self._mesh_renderers = (
            MeshRenderer(rasterizer=MeshRasterizer(raster_settings=heuristic), shader=shader).to(self.device),
            MeshRenderer(rasterizer=MeshRasterizer(raster_settings=naive), shader=shader).to(self.device),
        )
        return self._mesh_renderers

    @torch.no_grad()
    def render_views(self, points, elevs, azims, dist=2.5, fov_deg=50.0, cmap_name="viridis"):
        """points: (N,3) tensor. Returns (V,H,W,3) uint8."""
        assert len(elevs) == len(azims)
        points = points.to(self.device).float().unsqueeze(0)
        points = normalize_vertices(points)
        colors = self._height_colors(points, cmap_name)

        cloud = Pointclouds(points=points, features=colors).extend(len(elevs))
        rendered = self.renderer(cloud, cameras=self._cameras(elevs, azims, dist, fov_deg))
        return (rendered[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    @torch.no_grad()
    def render_mesh_views(self, verts, faces, elevs, azims, dist=2.5, fov_deg=50.0,
                          cmap_name="viridis"):
        """Mirrors render_views step for step - same normalisation, same colour
        ramp, same camera - so a mesh cell lands framed like a point cell."""
        assert len(elevs) == len(azims)
        verts = verts.to(self.device).float().unsqueeze(0)
        faces = faces.to(self.device).long().unsqueeze(0)
        verts = normalize_vertices(verts)
        colors = self._height_colors(verts, cmap_name)

        mesh = Meshes(verts=verts, faces=faces,
                      textures=TexturesVertex(verts_features=colors)).extend(len(elevs))
        cameras = self._cameras(elevs, azims, dist, fov_deg)

        heuristic, naive = self._build_mesh_renderers()
        try:
            rendered = heuristic(mesh, cameras=cameras)
        except RuntimeError:
            # Heuristic binning can overflow on pathological meshes.
            rendered = naive(mesh, cameras=cameras)
        return (rendered[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def render_single_view(self, points, elev, azim, **kwargs):
        return self.render_views(points, [elev], [azim], **kwargs)[0]

    def render_payload(self, payload, elev, azim, **kwargs):
        if isinstance(payload, MeshPayload):
            return self.render_mesh_views(payload.verts, payload.faces, [elev], [azim], **kwargs)[0]
        return self.render_views(payload, [elev], [azim], **kwargs)[0]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _render_row(plan, cls, obj_id, renderer, elev, azim, dist, load_opts, cmap_name):
    return [
        renderer.render_payload(m.load(cls, obj_id, load_opts), elev=elev, azim=azim,
                                dist=dist, cmap_name=cmap_name)
        for m in plan.columns
    ]


def make_comparison_grid_figure(
    *,
    plan: GridPlan,
    elev=20.0,
    azim=DEFAULT_FRONTAL_AZIM,
    dist=2.5,
    out_path=None,
    renderer=None,
    image_size=512,
    device=None,
    load_opts: LoadOpts = None,
    row_labels="auto",
    cmap_name="viridis",
    dpi=300,
    formats=("png", "pdf"),
    cell_size=2.6,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    renderer = renderer or FigureRenderer(device=device, image_size=image_size)
    load_opts = load_opts or LoadOpts()

    # Render first so a failed cell shrinks the grid instead of holing it. Files
    # can go missing or half-written while inference is still running.
    rendered_rows = []
    for cls, obj_id in plan.rows:
        queue = [obj_id] + list(plan.fallbacks.get(cls.taxonomy_id, []))
        for candidate in queue:
            try:
                imgs = _render_row(plan, cls, candidate, renderer, elev, azim, dist,
                                   load_opts, cmap_name)
            except Exception as exc:
                print(f"  skipping {cls.name}/{candidate}: {type(exc).__name__}: {exc}")
                continue
            rendered_rows.append((cls, candidate, imgs))
            if candidate in plan.fallbacks.get(cls.taxonomy_id, []):
                plan.fallbacks[cls.taxonomy_id].remove(candidate)
            break

    if not rendered_rows:
        raise RuntimeError("Every row failed to render.")

    n_rows, n_cols = len(rendered_rows), len(plan.columns)
    show_labels = row_labels == "class" or (
        row_labels == "auto" and len({c.taxonomy_id for c, _, _ in rendered_rows}) > 1
    )

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell_size * n_cols, cell_size * n_rows),
                             squeeze=False)
    for row, (cls, obj_id, imgs) in enumerate(rendered_rows):
        for col, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(plan.columns[col].display_name)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    if show_labels:
        # imshow forces aspect="equal", and apply_aspect() only runs at draw
        # time - so positions must be read after an explicit draw.
        fig.canvas.draw()
        blocks, start = [], 0
        for i in range(1, n_rows + 1):
            if i == n_rows or rendered_rows[i][0] != rendered_rows[start][0]:
                blocks.append((rendered_rows[start][0], start, i - 1))
                start = i
        for cls, first, last in blocks:
            p0 = axes[first, 0].get_position()
            p1 = axes[last, 0].get_position()
            # fig.text costs no grid cell, so the tight wspace/hspace is untouched.
            fig.text(p0.x0 - 0.012, (p0.y1 + p1.y0) / 2, cls.name.capitalize(),
                     rotation=90, va="center", ha="center", fontweight="bold", fontsize=20)

    out_path = out_path or f"{DEFAULT_OUT_DIR}/comparison_grid"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = []
    for ext in formats:
        p = f"{out_path}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


def make_per_class_figures(*, methods, classes, num_per_class, seed, out_dir,
                           prefix="comparison", ranker=None, **kw):
    written = []
    for cls in classes:
        try:
            plan = resolve_grid(methods, [cls], num_per_class, seed=seed, ranker=ranker)
        except RuntimeError as exc:
            print(f"  skipping {cls.name}: {exc}")
            continue
        written += make_comparison_grid_figure(
            plan=plan, out_path=f"{out_dir}/{prefix}_{cls.name}", **kw
        )
    return written


def make_input_multiview_figure(
    *,
    obj_id=None,
    cls: ClassSpec = None,
    methods=None,
    num_views=4,
    elev=20.0,
    dist=2.5,
    azim_start=DEFAULT_FRONTAL_AZIM,
    out_path=None,
    seed=0,
    renderer=None,
    image_size=512,
    device=None,
    cmap_name="viridis",
    dpi=300,
    formats=("png", "pdf"),
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    renderer = renderer or FigureRenderer(device=device, image_size=image_size)
    cls = cls or _BY_NAME["chair"]

    methods = methods or build_registry()
    inp = next((m for m in methods if m.kind == "input"), None)
    if inp is None or not inp.has_class(cls):
        raise RuntimeError(f"No input point clouds available for class '{cls.name}'.")

    index = inp.index(cls)
    if obj_id is None:
        obj_id = random.Random(seed).choice(sorted(index))
    elif obj_id not in index:
        raise SystemExit(f"Object '{obj_id}' has no input point cloud in {inp.class_dir(cls)}")

    points = inp.load(cls, obj_id, LoadOpts())
    azims = (np.linspace(0, 360, num_views, endpoint=False) + azim_start) % 360
    views = renderer.render_views(points, elevs=[elev] * num_views, azims=list(azims),
                                  dist=dist, cmap_name=cmap_name)
    col_names = FIXED_VIEW_NAMES.get(num_views,
                                     [f"View {i + 1} ({a:.0f}°)" for i, a in enumerate(azims)])

    fig, axes = plt.subplots(1, num_views, figsize=(3 * num_views, 3.3), squeeze=False)
    for ax, img, name in zip(axes[0], views, col_names):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(name)
    fig.tight_layout()

    out_path = out_path or f"{DEFAULT_OUT_DIR}/input_multiview"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = []
    for ext in formats:
        p = f"{out_path}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Generate paper-quality point cloud figures.")
    p.add_argument("--mode", choices=["input_views", "comparison", "both"], default="both")
    p.add_argument("--dry_run", action="store_true", help="Resolve and print the plan, render nothing.")

    p.add_argument("--classes", nargs="+", default=["all"],
                   help="Class names, taxonomy ids, or 'all'.")
    p.add_argument("--num_per_class", type=int, default=2)
    p.add_argument("--object_ids", nargs="+", default=None,
                   help="Pin exact rows as TAXONOMY:OBJECT_ID; bypasses sampling.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude_methods", nargs="+", default=[])

    p.add_argument("--select", choices=["random", "best", "uhd", "fscore"], default="best",
                   help="Row choice: sample at random, or keep the objects our model "
                        "reconstructs most faithfully - by both metrics, by UHD alone "
                        "(fidelity to the input), or by F-Score alone.")
    p.add_argument("--rank_method", type=str, default="lora64",
                   help="Which of our columns --select scores, e.g. lora64; "
                        "default is the highest rank available.")
    p.add_argument("--rank_pool", type=int, default=0,
                   help="Score only this many candidates per class (0 = all). "
                        "Scoring costs roughly a second per object.")

    p.add_argument("--universe", choices=list(UNIVERSES), default=DEFAULT_UNIVERSE,
                   help="pcn_test = PCN's own partials; custom_input = our generated scans.")
    p.add_argument("--pcn_root", type=str, default=str(DEFAULT_PCN_ROOT))
    p.add_argument("--pcn_split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--input_pc_root", type=str, default=str(DEFAULT_INPUT_PC_ROOT))
    p.add_argument("--shapenet_root", type=str, default=str(DEFAULT_SHAPENET_ROOT),
                   help="Source meshes; --select scores custom_input against these.")
    p.add_argument("--adapointr_root", type=str, default=str(DEFAULT_ADAPOINTR_ROOT))
    p.add_argument("--svdformer_root", type=str, default=str(DEFAULT_SVDFORMER_ROOT))
    p.add_argument("--lora_root", type=str, default=str(DEFAULT_LORA_ROOT))
    p.add_argument("--lora_ranks", nargs="+", type=int, default=list(DEFAULT_LORA_RANKS))

    p.add_argument("--render_ours_as_mesh", action="store_true",
                   help="Render our models' mesh.obj as a surface instead of sampled points.")
    p.add_argument("--mesh_samples", type=int, default=16384)
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=DEFAULT_FRONTAL_AZIM)
    p.add_argument("--dist", type=float, default=2.5)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--point_radius", type=float, default=0.008)
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--figure_per_class", action="store_true")
    p.add_argument("--row_labels", choices=["auto", "class", "none"], default="auto")
    p.add_argument("--cell_size", type=float, default=2.6)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])

    p.add_argument("--input_class", type=str, default="chair")
    p.add_argument("--object_id", type=str, default=None)
    p.add_argument("--num_views", type=int, default=4)
    return p.parse_args()


def main():
    args = _parse_args()
    classes = resolve_classes(args.classes)
    methods = build_registry(
        universe=args.universe,
        pcn_root=args.pcn_root,
        split=args.pcn_split,
        input_pc_root=args.input_pc_root,
        adapointr_root=args.adapointr_root,
        svdformer_root=args.svdformer_root,
        lora_root=args.lora_root,
        lora_ranks=args.lora_ranks,
        exclude=set(args.exclude_methods),
    )

    pinned = None
    if args.object_ids:
        pinned = {}
        for token in args.object_ids:
            tax, _, oid = token.partition(":")
            if not oid:
                raise SystemExit(f"--object_ids expects TAXONOMY:OBJECT_ID, got '{token}'")
            pinned.setdefault(tax, []).append(oid)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ours_keys = [m.key for m in methods if m.kind == "ours"]
    if args.rank_method and args.rank_method not in ours_keys:
        raise SystemExit(f"Unknown --rank_method '{args.rank_method}'. "
                         f"Ours columns: {', '.join(ours_keys) or 'none'}.")
    ranker = None if args.select == "random" or not ours_keys else FidelityRanker(
        universe=args.universe, ours_key=args.rank_method or ours_keys[-1],
        criterion=args.select, pool=args.rank_pool, device=device,
        shapenet_root=Path(args.shapenet_root) if args.universe == "custom_input" else None,
        cache=Path(args.out_dir) / "fidelity_scores.csv",
    )

    if args.dry_run:
        plan = resolve_grid(methods, classes, args.num_per_class, seed=args.seed,
                            pinned=pinned, ranker=ranker)
        print(describe_plan(plan))
        return

    renderer = FigureRenderer(device=device, image_size=args.image_size,
                              radius=args.point_radius)
    load_opts = LoadOpts(render_ours_as_mesh=args.render_ours_as_mesh,
                         mesh_samples=args.mesh_samples)
    render_kw = dict(elev=args.elev, azim=args.azim, dist=args.dist, renderer=renderer,
                     load_opts=load_opts, row_labels=args.row_labels, cmap_name=args.cmap,
                     dpi=args.dpi, formats=tuple(args.formats), cell_size=args.cell_size)

    if args.mode in ("input_views", "both"):
        written = make_input_multiview_figure(
            obj_id=args.object_id,
            cls=resolve_classes([args.input_class])[0],
            methods=methods,
            num_views=args.num_views,
            elev=args.elev,
            dist=args.dist,
            azim_start=args.azim,
            out_path=f"{args.out_dir}/input_multiview_{args.universe}",
            seed=args.seed,
            renderer=renderer,
            cmap_name=args.cmap,
            dpi=args.dpi,
            formats=tuple(args.formats),
        )
        print("Saved input multiview figure to " + " / ".join(written))

    if args.mode in ("comparison", "both"):
        if args.figure_per_class:
            written = make_per_class_figures(
                methods=methods, classes=classes, num_per_class=args.num_per_class,
                seed=args.seed, out_dir=args.out_dir, prefix=f"comparison_{args.universe}",
                ranker=ranker, **render_kw
            )
        else:
            plan = resolve_grid(methods, classes, args.num_per_class, seed=args.seed,
                                pinned=pinned, ranker=ranker)
            print(describe_plan(plan))
            written = make_comparison_grid_figure(
                plan=plan, out_path=f"{args.out_dir}/comparison_{args.universe}", **render_kw
            )
        print("Saved comparison figure(s) to " + " / ".join(written))


if __name__ == "__main__":
    main()
