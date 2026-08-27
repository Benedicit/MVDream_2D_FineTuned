
import os
import sys
import random
import warnings
from pathlib import Path

working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import numpy as np
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
from scripts.util import get_mesh_from_pc, count_label_entries, load_pcd_to_tensor
from view_renderer import PointRenderer, MeshRendererMVDream

working_dir = os.path.dirname(os.path.abspath(__file__))

class CacheDataset(Dataset):
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

class PCNDataset(Dataset):
    def __init__(self, cache, samples):
        self.cache = cache
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = self.cache[str(s)]
        # Return tensors directly for DataLoader collation
        return {
            "z": item["z"],
            "c_text": item["c_text"],
            "pc_feat": item["pc_feat"],
        }, s


def collate_pc_feat(batch):
    items, sample_ids = zip(*batch)
    pc_feats = [item["pc_feat"] for item in items]
    pc_feats = [x.unsqueeze(0) if x.dim() == 1 else x for x in pc_feats]
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

class PCNDataModule(L.LightningDataModule):
    def __init__(
            self,
            model,
            num_views: int = 4,
            H: int = 256,
            ELEV_DEG: float = 15.0,
            AZIM_START: float = 0.0,
            AZIM_SPAN: float = 360.0,
            # dataset params
            batch_size: int = 16,
            debug_dir: str = "debug/cache",
            cache_dir: str = "cache",
            use_pcn = False,
    ):
        super().__init__()
        self.num_views = num_views
        self.H = H
        self.ELEV_DEG = ELEV_DEG
        self.AZIM_START = AZIM_START
        self.AZIM_SPAN = AZIM_SPAN
        self.batch_size = batch_size
        self.debug_dir = debug_dir
        self.model = model

        self.train_cache = None
        self.val_cache = None
        self.cache_dir = cache_dir
        self.pcn_root = Path(f"{working_dir}/../../data/.pcn/ShapeNetCompletion/train/partial")
        self.shapenet_root = Path(f"{working_dir}/../../data/.shapenet/")

    def load_pcn(self,root_dir: str | Path) -> dict[str, torch.Tensor]:
        """
        Traverse partial/synset_id/*.pcd and return a dict mapping
        relative path -> (N, 3) tensor.
        """
        root = Path(root_dir)
        pcd_files = sorted(root.glob(f"**/*.pcd"))

        dataset = {}
        for pcd_path in pcd_files:
            rel_path = pcd_path.relative_to(root)
            dataset[str(rel_path)] = load_pcd_to_tensor(pcd_path)

        return dataset


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

        train_path = os.path.join(self.cache_dir, "train_cache_pcn.pt")

        if not os.path.exists(train_path):
            os.makedirs(self.cache_dir, exist_ok=True)
            # The objects are counted from 1 up
            self.build_cache( f"{self.debug_dir}/train", train_path)

        # Clean up
        del self._pc_encoder, self._renderer, self._camera
        torch.cuda.empty_cache()

    def setup(self, stage: str = None):
        """Called on every rank — just load from disk."""
        train_path = os.path.join(self.cache_dir, "train_cache_pcn.pt")

        if stage in ("fit", None):
            train_data = torch.load(train_path, map_location="cpu")
            self.train_cache   = train_data["cache"]
            self.train_samples = train_data["samples"]


    @torch.no_grad()
    def build_cache(self, debug_dir: str, train_path, save_target_imgs: bool = False, save_pc_imgs: bool = False) -> dict:

        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cache = {}
        obj_counter = {}
        obj_cache = {}
        paths = self.pcn_root.glob("**/*.pcd")
        paths = sorted(paths)
        samples = []
        pbar = tqdm(total=len(paths), desc="Building Cache", unit="samples")
        c_text = self.model.get_learned_conditioning([""])
        c_text = c_text.repeat(self.num_views, *([1] * (c_text.dim() - 1))).contiguous()
        print("Starting to build cache, this may take a while.")

        for pcd_path in paths:
            rel        = pcd_path.relative_to(self.pcn_root)  # synset_id/obj_id/00.pcd
            synset_id  = rel.parts[0]
            obj_id     = rel.parts[1]
            counter = obj_counter.get(obj_id, 0)
            if counter > 0:
                continue
            counter += 1
            obj_counter[obj_id] = counter

            obj_path = (
                    self.shapenet_root
                    / synset_id / obj_id
                    / "models" / "model_normalized.obj"
            )
            z = obj_cache.get(obj_id)
            points = load_pcd_to_tensor(pcd_path)
            if z is None:

                if obj_path.exists():
                    mesh = load_objs_as_meshes([Path(obj_path)], device=self.device, load_textures=False)
                    verts = mesh.verts_packed()
                    faces = mesh.faces_packed()
                else:
                    print(f"[warn] No mesh for {synset_id}/{obj_id}, skipping.")
                    continue
                target_imgs = self._renderer.render_mvdream_views(verts, faces, camera=self._camera).contiguous()
                z = self.model.encode_first_stage(target_imgs)
                if hasattr(self.model, "get_first_stage_encoding"):
                    z = self.model.get_first_stage_encoding(z)
                z = z.detach().contiguous()
                obj_cache[obj_id] = z.cpu()

            idx = torch.randperm(points.size(0))[:500]
            sparse_points = points[idx]
            feat, _ = self._pc_encoder(sparse_points)



            sample = f"{obj_id}_{pcd_path.stem}"
            samples.append(sample)

            cache[sample] = {
                "z": z.cpu(),
                "c_text": c_text.cpu(),
                "pc_feat": feat.squeeze(0).detach().cpu(),
            }
            
            pbar.set_description(f"Processed {sample}")
            pbar.update(1)

        torch.save({"cache": cache, "samples": samples}, train_path)

    def train_dataloader(self):

        return DataLoader(
            PCNDataset(self.train_cache, self.train_samples),
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
        pass

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
            use_pcn = False,
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
        for cl in self.class_names:
            for a in range(start, end):
                num_obj = count_label_entries(cl)
                # We will do an 70/15/15 split
                if a > num_obj * 0.7:
                    break
                samples.append(f"shapenet_{cl}{a}")
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

        if not os.path.exists(train_path):
            os.makedirs(self.cache_dir, exist_ok=True)
            # The objects are counted from 1 up
            train_samples = self._build_sample_list(1, self.num_samples + 1)
            train_cache = self.build_cache(train_samples, f"{self.debug_dir}/train")
            torch.save({"cache": train_cache, "samples": train_samples}, train_path)

        if not os.path.exists(val_path):
            """
            num_val = int(self.num_samples * 0.1)
            num_val += num_val % self.batch_size
            num_val = max(self.batch_size, num_val)
            """
            num_val = 96
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
            verts_m, faces_m, mesh_obj = self.get_verts_faces_from_mesh(mesh_path)
            target_imgs = self._renderer.render_mvdream_views(verts_m, faces_m, camera=self._camera).contiguous()

            points, normals = sample_points_from_meshes(mesh_obj, num_samples=8192, return_normals=True)
            points = points.squeeze(0)
            normals = normals.squeeze(0)
            N_pool, _ = points.shape
            offset = random.random() * 0.0
            percentage_kept = 0.75
            dropout_mask = torch.rand(N_pool, device=self.device) < (4096 / N_pool * percentage_kept)

            for split_axis, key in [(0, "pc_feat_x_split"), (2, "pc_feat_z_split")]:
                axis_mask = points[..., split_axis] > offset
                mask = axis_mask & dropout_mask
                if mask.sum() < 500:
                    axis_mask = points[..., 1] > offset
                    mask = axis_mask & dropout_mask
                    if mask.sum() < 500:
                        print("Splitting in half was not successful -> fallback to only random masking")
                        mask = torch.rand(N_pool, device=self.device) < 0.15
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
            pbar.set_description(f"Processed {sample}")
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

    def get_verts_faces_from_mesh(self, mesh_path=None, textures=False):
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
            mesh = load_objs_as_meshes([Path(mesh_path)], device=self.device, load_textures=textures)
            verts = mesh.verts_packed()
            faces = mesh.faces_packed()

        # Return renderer, verts, and faces (since the renderer now needs both)
        return verts, faces, mesh if mesh else None