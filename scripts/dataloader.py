
import os
import sys
import random
import warnings
from pathlib import Path

working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch3d.io import load_ply, load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from mvdream.model_zoo import build_model
from mvdream.camera_utils import get_camera
from scripts.pc_encoder import PointCloudEncoder
from scripts.pointnet_encoder import get_point_cloud_name_reg
from scripts.tester import get_mesh_from_pc
from view_renderer import PointRenderer, MeshRendererMVDream

working_dir = os.path.dirname(os.path.abspath(__file__))

class CacheDataset(Dataset):
    # TODO: Remove samples
    def __init__(self, cache, samples):
        self.cache = cache
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = self.cache[s]
        # Return tensors directly for DataLoader collation
        return {
            "z": item["z"],
            "c_text": item["c_text"],
            "pc_feat": item["pc_feat_x_split"] if random.random() > 0.5 else item["pc_feat_z_split"],
        }, s

def collate_pc_feat(batch):
    items, sample_ids = zip(*batch)
    pc_feats = [item["pc_feat"] for item in items]
    lengths = torch.tensor([x.shape[0] for x in pc_feats], dtype=torch.long)
    flat_pc_feat = torch.cat(pc_feats, dim=0)
    batch_idx = torch.repeat_interleave(
        torch.arange(len(pc_feats), device=flat_pc_feat.device),
        lengths,
    )
    pc_feat_dense, pc_feat_mask = to_dense_batch(flat_pc_feat, batch_idx)

    z = torch.stack([item["z"] for item in items], dim=0)
    c_text = torch.stack([item["c_text"] for item in items], dim=0)
    return {
        "z": z,
        "c_text": c_text,
        "pc_feat": pc_feat_dense,
        "pc_feat_mask": pc_feat_mask,
    }

class ShapeDreamDataModule(L.LightningDataModule):
    def __init__(
            self,
            class_names: list[str],
            model,
            num_views: int = 4,
            H: int = 256,
            ELEV_DEG: float = 15.0,
            AZIM_START: float = 0.0,
            AZIM_SPAN: float = 360.0,
            # dataset params
            num_samples: int = 4000,
            batch_size: int = 16,
            debug_dir: str = "debug/cache",
            cache_dir: str = "cache",
    ):
        super().__init__()
        self.class_names = class_names
        self.num_views = num_views
        self.H = H
        self.ELEV_DEG = ELEV_DEG
        self.AZIM_START = AZIM_START
        self.AZIM_SPAN = AZIM_SPAN
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.debug_dir = debug_dir
        self.model = model

        self.train_cache = None
        self.val_cache = None
        self.cache_dir = cache_dir

    def _build_sample_list(self, start: int, end: int) -> list[str]:
        samples = []
        for a in range(start, end):
            for cl in self.class_names:
                samples.append(f"shapenet_{cl}{a}.ply")
        return samples


    def prepare_data(self):
        """Called only on rank 0"""
        device = torch.device("cuda:0")

        pc_encoder = PointCloudEncoder().to(device)
        renderer   = MeshRendererMVDream(device=device, image_size=self.H)
        camera     = get_camera(
            num_frames=self.num_views,
            elevation=self.ELEV_DEG,
            azimuth_start=self.AZIM_START,
            azimuth_span=self.AZIM_SPAN,
            blender_coord=False,
        ).to(device)
        model = self.model.to(device)
        model.eval()

        # Temporarily assign so build_cache can use them via self
        self._pc_encoder = pc_encoder
        self._renderer   = renderer
        self._camera     = camera
        self.device      = device

        train_path = os.path.join(self.cache_dir, "train_cache.pt")
        val_path   = os.path.join(self.cache_dir, "val_cache.pt")

        os.makedirs(self.cache_dir, exist_ok=True)
        train_samples = self._build_sample_list(1, self.num_samples + 1)
        train_cache = self.build_cache(train_samples, f"{self.debug_dir}/train")
        torch.save({"cache": train_cache, "samples": train_samples}, train_path)

        if not os.path.exists(val_path):
            num_val = int(self.num_samples * 0.1)
            num_val += num_val % self.batch_size
            num_val = max(self.batch_size, num_val)
            val_start = self.num_samples + 50
            val_samples = self._build_sample_list(val_start, val_start + num_val)
            val_cache = self.build_cache(val_samples, f"{self.debug_dir}/val")
            torch.save({"cache": val_cache, "samples": val_samples}, val_path)
            print(f"[rank0] Val cache saved to {val_path}")
        else:
            print(f"[rank0] Val cache already exists, skipping.")

        # Clean up
        del self._pc_encoder, self._renderer, self._camera
        torch.cuda.empty_cache()

    def setup(self, stage: str = None):
        """Called on every rank — just load from disk."""
        train_path = os.path.join(self.cache_dir, "train_cache.pt")
        val_path   = os.path.join(self.cache_dir, "val_cache.pt")

        if stage in ("fit", None):
            train_data = torch.load(train_path, map_location="cpu")
            self.train_cache   = train_data["cache"]
            self.train_samples = train_data["samples"]

            val_data = torch.load(val_path, map_location="cpu")
            self.val_cache   = val_data["cache"]
            self.val_samples = val_data["samples"]

        if stage == "validate":
            val_data = torch.load(val_path, map_location="cpu")
            self.val_cache   = val_data["cache"]
            self.val_samples = val_data["samples"]

    @torch.no_grad()
    def build_cache(self, samples: list[str], debug_dir: str, save_target_imgs: bool = False, save_pc_imgs: bool = False) -> dict:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        text_cond_cache = {}
        cache = {}

        pbar = tqdm(total=len(samples), desc="Building Cache", unit="samples")
        for sample in samples:

            mesh_path = get_mesh_from_pc(sample)
            verts_m, faces_m, mesh_obj = self.get_renderings_verts_from_file_mesh(mesh_path)
            target_imgs = self._renderer.render_mvdream_views(verts_m, faces_m, camera=self._camera).contiguous()

            points, normals = sample_points_from_meshes(mesh_obj, num_samples=8192, return_normals=True)
            points = points.squeeze(0)
            normals = normals.squeeze(0)
            N_pool, _ = points.shape
            offset = random.random() * 0.015
            percentage_kept = 0.75
            dropout_mask = torch.rand(N_pool, device=self.device) < (4096 / N_pool * percentage_kept)

            for split_axis, key in [(0, "pc_feat_x_split"), (2, "pc_feat_z_split")]:
                axis_mask = points[..., split_axis] > offset
                mask = axis_mask & dropout_mask
                flat_points = points[mask].detach().cpu()
                flat_normals = normals[mask].detach().cpu()

                feat, _ = self._pc_encoder(flat_points)
                cache.setdefault(sample, {})[key] = feat.squeeze(0).detach().cpu()

            z = self.model.encode_first_stage(target_imgs)
            if hasattr(self.model, "get_first_stage_encoding"):
                z = self.model.get_first_stage_encoding(z)
            z = z.detach().contiguous()

            prompt = "a " + get_point_cloud_name_reg(sample)
            c1 = text_cond_cache.get(prompt)
            if c1 is None:
                c1 = self.model.get_learned_conditioning([prompt])
                c1 = c1.detach().contiguous()
                text_cond_cache[prompt] = c1
            c_text = c1.repeat(self.num_views, *([1] * (c1.dim() - 1))).contiguous()

            cache[sample].update({
                "z": z.cpu(),
                "c_text": c_text.cpu(),
            })
            pbar.update(1)

        return cache

    def train_dataloader(self):

        return DataLoader(
            CacheDataset(self.train_cache, self.train_samples),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_pc_feat,
        )

    def val_dataloader(self):
        return DataLoader(
            CacheDataset(self.val_cache, self.val_samples),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_pc_feat,
        )

    def get_renderings_verts_from_file_mesh(self, mesh_path=None, textures=False):
        """
        mesh_path: Path to a .ply or .obj file containing mesh data
        """
        mesh = None
        # 1. Load vertices and faces
        if mesh_path.endswith(".ply"):
            verts, faces = load_ply(mesh_path)
            verts = verts.to(self.device)
            faces = faces.to(self.device)
        else:
            # For .obj files, load_objs_as_meshes is often more robust
            mesh = load_objs_as_meshes([mesh_path], device=self.device, load_textures=textures)
            verts = mesh.verts_packed()
            faces = mesh.faces_packed()

        # Return renderer, verts, and faces (since the renderer now needs both)
        return verts, faces, mesh if mesh else None