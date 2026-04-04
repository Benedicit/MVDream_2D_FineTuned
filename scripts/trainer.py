import math
import os
from pathlib import Path

import pytorch3d
import torch
import random
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image as PilImage 
import numpy as np

from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.ops import sample_points_from_meshes

from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.interface import LatentDiffusionInterface
from mvdream.camera_utils import get_camera
from mvdream.model_zoo import build_model
from lora import add_lora_to_cross_att_only, add_lora_to_attention_and_conv, add_lora_to_all_layers, LoRAConv2d, LoRALinear
from pointnet_encoder import read_from_plyfile, get_pointnet_features, PointFeatProjector, get_point_cloud_name, get_point_cloud_name_reg
from view_renderer import PointRenderer, MeshRendererMVDream
from tqdm import tqdm
from tester import Tester3D
import pandas as pd
from tester import Tester3D
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from pc_encoder import PointCloudEncoder

working_dir = os.path.dirname(os.path.abspath(__file__))

from yanx_pointnet2_encoder import YanxPointNet2Encoder
from tester import Tester3D
from flow_matching import FlowMatching

SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = f"{working_dir}/../debug"
MESH_DIR = f"{working_dir}/../debug_3D"

gso_csv = f"{working_dir}/../../data/gso_label_to_mesh.csv"
shapenet_csv = f"{working_dir}/../../data/shapenet_label_to_mesh.csv"
mapping_gso = pd.read_csv(gso_csv)
mapping_shapenet = pd.read_csv(shapenet_csv)


def get_mesh_from_pc(pointcloud_name="bag1.ply"):
    if pointcloud_name.startswith("shapenet"):
        return mapping_shapenet.loc[mapping_shapenet["label"] == pointcloud_name, "filename"].iloc[0]
    return mapping_gso.loc[mapping_gso["label"] == pointcloud_name, "filename"].iloc[0]

def save_training_views_grid(imgs, out_path, pad=16):
    """
    imgs: (V,3,H,W) in [-1,1]
    """
    imgs_np = (0.5 * (imgs + 1.0)).clamp(0,1)
    imgs_np = (imgs_np.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    V, H, W, C = imgs_np.shape
    canvas_h = H + 2 * pad
    canvas_w = V * W + (V + 1) * pad
    canvas = np.zeros((canvas_h, canvas_w, C), dtype=np.uint8)

    y = pad
    for i in range(V):
        x = pad + i * (W + pad)
        canvas[y:y + H, x:x + W, :] = imgs_np[i]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    PilImage.fromarray(canvas).save(out_path)
    print("Saved training views grid to", out_path)

def make_gt_of_sample_list(tester : Tester3D, samples, elev_deg=15.0, debug_dir: str = "ground_truth/", save_grid=False, save_4_views=True, generate_3D=False):
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    V = 4
    renderer = MeshRendererMVDream(device=device, image_size=256)
    # get camera once
    cam_gpu = get_camera(
        num_frames=V,
        elevation=elev_deg,
        azimuth_start=0.0,
        azimuth_span=360.0,
        blender_coord=False,
    ).to(device, non_blocking=True)
    cam = cam_gpu.contiguous()

    pbar = tqdm(total=len(samples), desc="Rendering GT meshes", unit="samples")
    for sample in samples:
        name = get_point_cloud_name_reg(sample, with_number=True)

        mesh_path = get_mesh_from_pc(sample)

        mesh = load_objs_as_meshes([mesh_path], device=device, load_textures=False)
        verts_m = mesh.verts_packed()
        faces_m = mesh.faces_packed()

        x = renderer.render_mvdream_views(verts_m, faces_m, camera=cam)


        if save_grid:
            save_training_views_grid(
                imgs=x,
                out_path=os.path.join(debug_dir, f"{name}_target.png"),
            )
        elif save_4_views:
            x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
            target_imgs = (x * 255.0).permute(0, 2, 3, 1).cpu().numpy()
            obj_path = debug_dir + name
            tester.save_4_views(target_imgs,obj_path)
            if generate_3D:
                tester.views_to_3D(obj_path)
        pbar.update(1)

class LoRATrainer:
    def __init__(self,
                 device,
                 lora_rank,
                 lora_alpha,
                 flow_matching=True,
                 start_from_noise=True,
                 model_name="sd-v2.1-base-4view",
                 H=256,
                 W=256,
                 ELEV_DEG=15.0,
                 DIST=2.5,
                 AZIM_START=0.0,
                 AZIM_SPAN=360.0,
                 num_views=4,
                 load_from_ckpth: bool = False,
                 ckpt_path=""):
        #torch.compiler.reset()
        self.device = torch.device(device)
        self.scaler = torch.amp.GradScaler('cuda')
        self.model = build_model(model_name=model_name)

        self.unet = self.model.model.diffusion_model
        self.flow_matching = flow_matching
        self.start_from_noise = start_from_noise

        if flow_matching:
            add_lora_to_all_layers(self.unet, r=lora_rank, alpha=lora_alpha)
        else:
            add_lora_to_cross_att_only(self.unet, r=lora_rank, alpha=lora_alpha)


        self.model.to(self.device)
        self.unet.to(self.device)

        self.model.device = self.device

        for p in self.model.parameters():
            p.requires_grad_(False)

        # Define lora_params
        self.lora_params = []
        lora_layer_count = 0
        for m in self.unet.modules():
            if isinstance(m, LoRALinear):
                m.base.weight.requires_grad_(False)
                if m.base.bias is not None:
                    m.base.bias.requires_grad_(False)
                m.lora_down.weight.requires_grad_(True)
                m.lora_up.weight.requires_grad_(True)
                self.lora_params.append(m.lora_down.weight)
                self.lora_params.append(m.lora_up.weight)
                lora_layer_count += 1

        # Define projector properly
        dummy_c = self.model.get_learned_conditioning(["dummy"]).to(self.device)
        self.dummy_pointcloud_path = f"{working_dir}/../../data/dataset_masked/bag1.ply"

        self.pointnet = YanxPointNet2Encoder(
        normal_channel=False,
        out_dim=256,
        device=device,
        )

        with torch.no_grad():
            pc_feat_dummy = get_pointnet_features(self.pointnet, pointcloud_path=self.dummy_pointcloud_path, device=self.device)
        pc_feat_dim = pc_feat_dummy.shape[-1]
        context_dim = dummy_c.shape[-1]
        #self.depth_map_encoder = DepthViTTokenEncoder(token_dim=context_dim, tokens_per_view=1, num_views=4).to(self.device)

        projector = PointFeatProjector(
                in_dim=pc_feat_dim,
                context_dim=context_dim,
                num_tokens=4,
        ).to(self.device)


        for p in projector.parameters():
            p.requires_grad_(True)
        self.projector = projector

        self.pc_encoder = PointCloudEncoder()
        for p in self.pc_encoder.projector.parameters():
            p.requires_grad_(True)

        lora_param_list = list(self.lora_params)
        print(f"LoRA parameters: {sum(p.numel() for p in lora_param_list)/1e6:.2f}, LoRA Layers: {lora_layer_count}")

        # Define optimizer
        #self.optimizer = torch.optim.AdamW( list(self.lora_params) + list(self.projector.parameters()) + list(self.depth_map_encoder.proj.parameters()), lr=1e-4,)
        self.optimizer = torch.optim.AdamW( lora_param_list + list(self.projector.parameters()) + list(self.pc_encoder.projector.parameters()) , lr=1e-4,)

        self.H = H
        self.W = W
        self.ELEV_DEG = ELEV_DEG
        self.AZIM_START = AZIM_START
        self.AZIM_SPAN = AZIM_SPAN
        self.DIST = DIST
        self.num_views = num_views

        self.renderer = MeshRendererMVDream(device=self.device, image_size=self.H)
        #self.tester = Tester3D()

        torch.manual_seed(42)
        self.ckpt_path = ckpt_path
        # check if depth_map_encoder already saved, else it will fail
        if load_from_ckpth:
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            self.unet.load_state_dict(ckpt["unet"], strict=False)
            self.projector.load_state_dict(ckpt["projector"], strict=True)
            #self.depth_map_encoder.load_state_dict(ckpt["depth_map_encoder"], strict=False)

        self.pytorch3d_io = pytorch3d.io.IO()

        if self.flow_matching:
            #add_lora_to_attention_and_conv(self.unet, r=lora_rank, alpha=lora_alpha)
            self.sampler = FlowMatching(self.device, self.model)
            self.compiled_loss = torch.compile(self._compute_loss_flow_matching, dynamic=False, disable=True)
            self.compiled_wrapper = torch.compile(self.training_wrapper, mode="max-autotune", dynamic=False,)
            # TODO: Split compile in pc-encoding and mvdream finetuning
            #self.compiled_loss = torch.compile(self._compute_loss_flow_matching, mode="max-autotune", dynamic=False,)
        else:
            self.sampler = DDIMSampler(self.model)
            self.compiled_loss = torch.compile(self._compute_loss_diffusion, mode="max-autotune", dynamic=False)

    def training_wrapper(self, x1, x0, cond, t):
        return self.sampler.training_losses(x1=x1, x0=x0, cond=cond, t=t)

    def _compute_loss_flow_matching(self, z_noisy, z, pc, t, c_text_flat, pc_feat_flat, camera_flat, V, noise):
        assert self.flow_matching

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            #print(f"Point cloud latent shape: {pc_latent.shape}", f"Target latent shape: {z.shape}")
            pc_tokens = self.projector(pc_feat_flat)
            if random.random() < 0.1:
                pc_tokens = torch.zeros_like(pc_tokens)

            context = torch.cat([pc_tokens], dim=1)
            cond = {"context": context, "camera": camera_flat, "num_frames": V}
            if self.start_from_noise:
                loss = self.sampler.training_losses(x1=z, x0=None, cond=cond, t=t)
            else:
                pc_latent = self.pc_encoder(coords=pc["points"], normals=pc["normals"], batch_lengths=pc["batch_lengths"])
                # add some noise to make it more random and as regularization
                noise = torch.randn_like(pc_latent) * 0.15

                #loss = self.sampler.training_losses(x1=z, x0=pc_latent + noise, cond=cond, t=t)
                loss = self.compiled_wrapper(x1=z, x0=pc_latent + noise, cond=cond, t=t)

                # KL-like Regularization: ensure pc_latent distribution matches N(0, 1)
                latent_mean = torch.mean(pc_latent)
                latent_std = torch.std(pc_latent)

                reg_loss = latent_mean ** 2 + (latent_std - 1) ** 2

                # Add regularization with a small weight
                loss = loss + 0.1 * reg_loss


        self.optimizer.zero_grad(set_to_none=True)
        # Scaled backward pass, theoretically not needed for BF16
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss


    def _compute_loss_diffusion(self, z_noisy, z, pc, t, c_text_flat, pc_feat_flat, camera_flat, V, noise):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pc_tokens = self.projector(pc_feat_flat)
            #context = torch.cat([c_text_flat, pc_tokens], dim=1)
            #context = torch.cat([c_text_flat], dim=1)
            context = torch.cat([pc_tokens], dim=1)
            cond = {"context": context, "camera": camera_flat, "num_frames": V}
            eps_pred = self.model.apply_model(z_noisy, t, cond)
            loss = F.mse_loss(eps_pred, noise)
        self.optimizer.zero_grad(set_to_none=True)
        # Scaled backward pass, theoretically not needed for BF16
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def get_renderings_verts_from_file_pc(self, pointcloud_path=None):
        """
        pointcloud_path: Path to pointcloud
        """
        # Get full pointcloud to train against
        points_obj = read_from_plyfile(pointcloud_path)
        verts = torch.tensor(points_obj, dtype=torch.float32, device=self.device)[:, :3]

        renderer = PointRenderer(device=self.device, image_size=self.H, radius=0.015)
        
        return renderer, verts

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


        # 3. Initialize the Mesh Renderer

        # Return renderer, verts, and faces (since the renderer now needs both)
        return verts, faces, mesh if mesh else None

    def save_weights(self, ckpt_path="checkpoints/mvdream_lora_pc_bag1_multiview.pt"):
        """
        save the weights in specified file
        self: Description
        """
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        #torch.save({"unet": self.unet.state_dict(), "projector": self.projector.state_dict(), "depth_map_encoder": self.depth_map_encoder.state_dict()}, ckpt_path)
        torch.save(
            {
            "unet": self.unet.state_dict(),
            "projector": self.projector.state_dict(),
            "pc_encoder" : self.pc_encoder.state_dict()
            }, ckpt_path)
        print("Saved", ckpt_path)

    @torch.no_grad()
    def build_cache(
        self,
        train_samples,
        base_path_masked: str,
        save_target_imgs: bool = False,
        save_pc_imgs: bool = False,
        debug_dir: str = "debug/cache",
    ):
        """
        Precompute and cache per-sample tensors so training can interleave samples cheaply.

        Returns:
            cache: dict[sample_name -> dict with keys:
                - "pc_path": str
                - "name": str
                - "pc_feat": (1,C) float tensor (on CPU)
                - "z": (V,4,h,w) float tensor (on CPU)
                - "camera": (V,4,4) float tensor (on CPU)
        """
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

        device = self.device
        V = int(self.num_views)

        # get camera once
        cam_gpu = get_camera(
            num_frames=V,
            elevation=self.ELEV_DEG,
            azimuth_start=0.0,
            azimuth_span=360.0,
            blender_coord=False,
        ).to(device, non_blocking=True)
        cam = cam_gpu.contiguous()
        cam_cpu = cam_gpu.detach().to("cpu", non_blocking=False).contiguous()


        # Put model(s) into eval once.
        self.model.eval()
        if hasattr(self, "pointnet") and self.pointnet is not None:
            self.pointnet.eval()

        text_cond_cache = {}

        cache = {}
        pbar = tqdm(total=len(train_samples), desc="Building Cache", unit="samples")
        for sample in train_samples:
            pc_path = os.path.join(base_path_masked, sample)
            name = get_point_cloud_name_reg(sample, with_number=True)

            # --- PointNet++ feature (likely dominates if it loads from disk) ---
            # TODO: Replace with PointCloud -> CLIP Encoder
            pc_feat = get_pointnet_features(self.pointnet, pointcloud_path=pc_path, device=device)
            if pc_feat.dim() == 1:
                pc_feat = pc_feat.unsqueeze(0)
            pc_feat = pc_feat.detach().contiguous()

            # --- Render target multi-view images ---
            mesh_path = get_mesh_from_pc(sample)
            verts_m, faces_m, mesh_obj = self.get_renderings_verts_from_file_mesh(mesh_path)
            target_imgs = self.renderer.render_mvdream_views(verts_m, faces_m, camera=cam).contiguous()

            with torch.no_grad():
                sampled_points, sampled_normals = sample_points_from_meshes(mesh_obj, num_samples=8192, return_normals=True)
                sampled_points = sampled_points.squeeze(0).cpu() # (8192, 3)
                sampled_normals = sampled_normals.squeeze(0).cpu() # (8192, 3)

            if save_target_imgs:
                save_training_views_grid(
                    imgs=target_imgs,
                    out_path=os.path.join(debug_dir, f"{name}_target.png"),
                )

            # --- Encode first stage ---
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = self.model.encode_first_stage(target_imgs)
                if hasattr(self.model, "get_first_stage_encoding"):
                    z = self.model.get_first_stage_encoding(z)
            z = z.detach().contiguous()

            # --- Optional debug rendering of the partial point cloud views ---
            pc_renderer, verts_pc = self.get_renderings_verts_from_file_pc(pc_path)
            pc_imgs = pc_renderer.render_mvdream_views(verts_pc, camera=cam)
            if save_pc_imgs:
                save_training_views_grid(
                    imgs=pc_imgs,
                    out_path=os.path.join(debug_dir, f"{name}_pc.png"),
                )

            # --- Text conditioning ---
            prompt = "a " + get_point_cloud_name_reg(sample)
            c1 = text_cond_cache.get(prompt)
            if c1 is None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    c1 = self.model.get_learned_conditioning([prompt])  # (1, ..., C) typically
                c1 = c1.detach().contiguous()
                text_cond_cache[prompt] = c1
            # Repeat along batch/view dimension
            c_text = c1.repeat(V, *([1] * (c1.dim() - 1))).contiguous()
            pbar.set_description(f"sample: {sample}")
            pbar.update(1)
            # --- Move cached tensors to CPU ---
            cache[sample] = {
                "pc_path": pc_path,
                "pc" : verts_pc,
                #"name": name,
                "training_points": sampled_points,
                "training_normals": sampled_normals,
                "pc_feat": pc_feat.to("cpu", non_blocking=False),
                "z": z.to("cpu", non_blocking=False),
                "camera": cam_cpu,  # shared if fixed
                #"V": V,
                #"target_imgs": target_imgs.detach().to("cpu", non_blocking=False),
                "c_text": c_text.to("cpu", non_blocking=False),
            }

        return cache

    def train_one_step_from_cache(self, item: dict):
        """
        One optimizer update using a cached sample dict produced by build_cache().

        Args:
            item: cache[sample] entry
        Returns:
            loss_value (float)
        """
        self.model.train()
        self.projector.train()
        self.pc_encoder.train()

        #V = int(item.get("V", self.num_views))

        z = item["z"].to(self.device).contiguous()
        camera = item["camera"].to(self.device).contiguous()
        pc_feat = item["pc_feat"].to(self.device).contiguous()
        pc_paths = item["pc_path"]
        points = item["training_points"].to(self.device).contiguous()
        normals = item["training_normals"].to(self.device).contiguous()


        c_text = item["c_text"].to(self.device).contiguous()

        B, V, C_lat, H_lat, W_lat = z.shape

        point_list = []
        normal_list = []
        lengths = []

        # --- Vectorized Augmentation ---
        B_pts, N_pool, _ = points.shape
        split_axis = 0 if random.random() < 0.5 else 2
        offset = random.random() * 0.015 - 0.01
        percentage_kept = 0.75

        # Compute masks for the whole batch at once
        axis_mask = points[..., split_axis] > offset
        dropout_mask = torch.rand((B_pts, N_pool), device=self.device) < (4096 / N_pool * percentage_kept)
        combined_mask = axis_mask & dropout_mask # [B, N]

        # Flatten to filter efficiently
        flat_points = points[combined_mask]
        flat_normals = normals[combined_mask]

        # Calculate lengths per batch element without a loop
        lengths = combined_mask.sum(dim=1).cpu().tolist()


        point_clouds = {
            "points": flat_points,
            "normals": flat_normals,
            "batch_lengths": lengths
        }

        # Flatten B and V for the UNet: (B*V, ...)
        z_flat = z.view(B * V, C_lat, H_lat, W_lat)
        camera_flat = camera.view(B * V, 16)
        c_text_flat = c_text.view(B * V, c_text.shape[-2], c_text.shape[-1])

        if self.flow_matching:
            t = torch.rand((B,), device=self.device)
        else:
            t = torch.randint(0, self.model.num_timesteps, (B,), device=self.device)
        t = t.repeat_interleave(V)  # (B*V,) each object in batch gets same t per view

        noise = torch.randn_like(z_flat)

        if not self.flow_matching:
            z_noisy = self.model.q_sample(z_flat, t, noise)
        else:
            z_noisy = z_flat
        pc_feat_flat = pc_feat.repeat_interleave(V, dim=0).contiguous()

        loss = self.compiled_loss(z_noisy, z_flat, point_clouds, t, c_text_flat, pc_feat_flat, camera_flat, V, noise)

        return float(loss.item())

    @torch.no_grad()
    def validation_step(
            self,
            val_cache: dict,
            num_samples: int = None,
            steps: int = 30,
            cfg_scale: float = 7.5,
            seed: int = 0,
    ) -> dict:
        """
        Run a full inference pass on validation samples and compare against
        the cached ground-truth images decoded from the VAE.

        Args:
            val_cache:   dict produced by build_cache()
            num_samples: if set, evaluate only this many randomly chosen samples
            steps:       number of ODE/DDIM steps for generation
            cfg_scale:   classifier-free guidance scale
            seed:        RNG seed for reproducible noise

        Returns:
            dict with keys "mse" ("psnr" (mean over samples))
        """
        self.model.eval()

        samples = list(val_cache.keys())
        if num_samples is not None:
            samples = random.sample(samples, min(num_samples, len(samples)))

        total_mse = 0.0
        total_psnr = 0.0

        for sample in samples:
            item = val_cache[sample]

            pc_feat   = item["pc_feat"].to(self.device).contiguous()   # (1, D)
            pc_latent = item["pc_latent"].to(self.device).contiguous() # (V, C, h, w)
            z_gt      = item["z"].to(self.device).contiguous()         # (1, V, C, h, w) or (V, C, h, w)
            V         = int(item["V"])

            # --- decode ground-truth latents to pixel space ---
            z_gt_flat = z_gt.view(V, *z_gt.shape[-3:])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                gt_pixels = self.model.decode_first_stage(z_gt_flat)   # (V, 3, H, W) in [-1,1]

            gt_pixels = torch.clamp((gt_pixels + 1.0) / 2.0, 0.0, 1.0)
            gt_pixels = (gt_pixels * 255.0).permute(0, 2, 3, 1).to(torch.float32)

            # --- generate images from the point cloud ---
            torch.manual_seed(seed)
            pc_feat_single = pc_feat[0] if pc_feat.ndim == 3 else pc_feat  # (1, D)
            pc_latent_views = pc_latent.view(V, *pc_latent.shape[-3:])

            gen_pixels = self.sample_multiview(
                pc_feat=pc_feat_single,
                pc_latent=pc_latent_views,
                use_pointcloud=True,
                num_views=V,
                H=self.H,
                W=self.W,
                steps=steps,
                scale=cfg_scale,
            )  # (V, H, W, 3) uint8


            # --- pixel-space metrics ---
            mse = F.mse_loss(gen_pixels, gt_pixels).item()
            total_mse += mse
            """
            # PSNR in [0,1] range: convert from [-1,1] first
            gt_01  = (gt_pixels  + 1.0) / 2.0
            gen_01 = (gen_pixels + 1.0) / 2.0
            mse_01 = F.mse_loss(gen_01, gt_01).item()
            psnr   = 10.0 * math.log10(1.0 / (mse_01 + 1e-8))
            total_psnr += psnr
            """

        n = len(samples)
        #return {"mse": total_mse / n, "psnr": total_psnr / n}
        return {"mse": total_mse / n}


    @torch.no_grad()
    def sample_multiview(
            self,
            pc_feat: torch.Tensor,
            pc_latent: torch.Tensor,
            prompt: str = "an object",
            use_pointcloud: bool = True,
            num_views: int = 4,
            H: int = 256,
            W: int = 256,
            steps: int = 100,
            scale: float = 7.5,
    ):
        """
        Sample num_views images from MVDream with or without PointNet++ conditioning.
        Returns [V, H, W, 3] uint8 numpy.
        """


        self.model.eval()

        latent_shape = [4, H // 8, W // 8]
        batch_size = num_views

        self.camera = get_camera(
            num_frames=num_views,
            elevation=self.ELEV_DEG,
            azimuth_start=self.AZIM_START,
            azimuth_span=self.AZIM_SPAN,
            blender_coord=False,
        ).to(self.device)

        if use_pointcloud:

            pc_feats_views = pc_feat.expand(num_views, -1)                         # [V,D_pc]
            pc_tokens = self.projector(pc_feats_views)                                  # [V,K,C]
            cond_context = torch.cat([pc_tokens], dim=1).to(self.device)         # [V,K,C]

            uc_pc_tokens = torch.zeros_like(pc_tokens, device=self.device)
            uc_context = torch.cat([uc_pc_tokens], dim=1).to(self.device)                    # [V,L+K,C]

        else:
            c_text = self.model.get_learned_conditioning([prompt] * num_views).to(self.device)   # [V,L,C]
            uc_text = self.model.get_learned_conditioning([""] * num_views).to(self.device)     # [V,L,C]
            cond_context = c_text                                                  # [V,L,C]
            uc_context = uc_text                                                   # [V,L,C]

        cond = {
            "context": cond_context,
            "camera": self.camera,
            "num_frames": num_views,
        }

        uc = {
            "context": uc_context,
            "camera": self.camera,
            "num_frames": num_views,
        }
        if self.flow_matching:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                args = {
                    "num_steps" : steps,
                    "cfg_scale" : scale,
                    "cond" : cond,
                    "uc_cond" : uc,
                }
                if self.start_from_noise:
                    x_source = torch.randn_like(pc_latent)
                x_source = x_source.to(self.device)
                samples = self.sampler.generate(x=x_source, sample_kwargs=args)
        else:
            with torch.amp.autocast("cuda", torch.bfloat16):
                samples, _ = self.sampler.sample(
                    S=steps,
                    conditioning=cond,
                    batch_size=batch_size,
                    shape=latent_shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=0.0,
                    x_T=None,
                )
        x = self.model.decode_first_stage(samples)
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        x = (x * 255.0).permute(0, 2, 3, 1)

        return x.to(torch.float32)