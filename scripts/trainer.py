import os
import random
from pathlib import Path

import numpy as np
import pytorch3d
import torch
import torch.nn.functional as F
from PIL import Image as PilImage
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

from lora import add_lora_to_cross_att_only, add_lora_to_all_layers, LoRALinear
from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from pc_encoder import PointCloudEncoder, PointCloudTransformer, PointCloudTransformerSmall
from pointnet_encoder import read_from_plyfile, get_pointnet_features, PointFeatProjector, get_point_cloud_name_reg
from view_renderer import PointRenderer, MeshRendererMVDream

working_dir = os.path.dirname(os.path.abspath(__file__))

from yanx_pointnet2_encoder import YanxPointNet2Encoder
from tester import Tester3D, get_mesh_from_pc
from flow_matching import FlowMatching

SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = f"{working_dir}/../debug"
MESH_DIR = f"{working_dir}/../debug_3D"

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
                 ckpt_path="",
                 no_compile=False):
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


        self.pc_encoder = PointCloudEncoder()

        #self.projector = PointCloudTransformer(n_layers=4)
        self.projector = PointCloudTransformerSmall(n_self_attn_layers=2, num_tokens=4)
        for p in self.projector.parameters():
            p.requires_grad_(True)

        lora_param_list = list(self.lora_params)
        print(f"LoRA parameters: {sum(p.numel() for p in lora_param_list)/1e6:.2f}M, LoRA Layers: {lora_layer_count}")
        print(f"Projector parameters: {sum(p.numel() for p in self.projector.parameters())/1e6:.3f}M")

        # Define optimizer
        #self.optimizer = torch.optim.AdamW( list(self.lora_params) + list(self.projector.parameters()) + list(self.depth_map_encoder.proj.parameters()), lr=1e-4,)
        self.optimizer = torch.optim.AdamW( lora_param_list + list(self.projector.parameters()), lr=1e-4,)

        self.H = H
        self.W = W
        self.ELEV_DEG = ELEV_DEG
        self.AZIM_START = AZIM_START
        self.AZIM_SPAN = AZIM_SPAN
        self.DIST = DIST
        self.num_views = num_views

        self.renderer = MeshRendererMVDream(device=self.device, image_size=self.H)
        #self.tester = Tester3D()


        self.camera = get_camera(
            num_frames=num_views,
            elevation=self.ELEV_DEG,
            azimuth_start=self.AZIM_START,
            azimuth_span=self.AZIM_SPAN,
            blender_coord=False,
        ).to(self.device)

        self.ckpt_path = ckpt_path
        # check if depth_map_encoder already saved, else it will fail
        if load_from_ckpth:
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            self.unet.load_state_dict(ckpt["unet"], strict=False)
            self.projector.load_state_dict(ckpt["projector"], strict=True)
            #self.depth_map_encoder.load_state_dict(ckpt["depth_map_encoder"], strict=False)

        self.pytorch3d_io = pytorch3d.io.IO()
        self.projector_fwd = torch.compile(self.projector, dynamic=True, disable=no_compile)

        if self.flow_matching:
            #add_lora_to_attention_and_conv(self.unet, r=lora_rank, alpha=lora_alpha)
            self.sampler = FlowMatching(self.device, self.model)
            self.compiled_loss = self._compute_loss_flow_matching
            self.compiled_wrapper = torch.compile(self.training_wrapper, mode="max-autotune", dynamic=False, disable=no_compile)

            #self.compiled_loss = torch.compile(self._compute_loss_flow_matching, mode="max-autotune", dynamic=False,)
        else:
            self.sampler = DDIMSampler(self.model)
            self.compiled_loss = torch.compile(self._compute_loss_diffusion, mode="max-autotune", disable=no_compile)

    def training_wrapper(self, x1, x0, cond, t):
        return self.sampler.training_losses(x1=x1, x0=x0, cond=cond, t=t)

    def _compute_loss_flow_matching(self, z_noisy, z, t, c_text_flat, pc_feat, pc_feat_mask, camera_flat, V, noise):
        assert self.flow_matching

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            #print(f"Point cloud latent shape: {pc_latent.shape}", f"Target latent shape: {z.shape}")

            cameras = camera_flat.view(camera_flat.shape[0] // 4, 4, -1)
            pc_tokens, pc_latent = self.projector_fwd(pc_feat, pc_feat_mask, cameras)

            if random.random() < 0.1:
                pc_tokens = torch.zeros_like(pc_tokens)

            context = torch.cat([pc_tokens], dim=1)
            cond = {"context": context, "camera": camera_flat, "num_frames": V}

            if self.start_from_noise:
                #loss = self.sampler.training_losses(x1=z, x0=None, cond=cond, t=t)
                loss = self.compiled_wrapper(x1=z, x0=None, cond=cond, t=t)
            else:
                # add some noise to make it more random and as regularization
                noise = torch.randn_like(pc_latent) * 0.2

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


    def _compute_loss_diffusion(self, z_noisy, z, t, c_text_flat, pc_feat, pc_feat_mask, camera_flat, V, noise):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pc_tokens = self.projector(pc_feat)
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

        renderer = PointRenderer(device=self.device, image_size=self.H, radius=0.015)
        
        return renderer

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
            #"pc_encoder" : self.pc_encoder.state_dict(),
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

        V = int(self.num_views)

        # Put model(s) into eval once.
        self.model.eval()

        text_cond_cache = {}

        cache = {}
        pbar = tqdm(total=len(train_samples), desc="Building Cache", unit="samples")
        for sample in train_samples:
            pc_path = os.path.join(base_path_masked, sample)
            name = get_point_cloud_name_reg(sample, with_number=True)

            # --- Render target multi-view images ---
            mesh_path = get_mesh_from_pc(sample)
            verts_m, faces_m, mesh_obj = self.get_renderings_verts_from_file_mesh(mesh_path)
            target_imgs = self.renderer.render_mvdream_views(verts_m, faces_m, camera=self.camera).contiguous()

            with torch.no_grad():
                points, normals = sample_points_from_meshes(mesh_obj, num_samples=8192, return_normals=True)
                # --- Vectorized Augmentation ---
                points = points.squeeze(0)
                normals = normals.squeeze(0)
                N_pool, _ = points.shape
                split_axis_zero = 0
                split_axis_two = 2
                offset = random.random() * 0.015
                percentage_kept = 0.75

                # Compute masks for the whole batch at once
                axis_mask = points[..., split_axis_zero] > offset
                dropout_mask = torch.rand(N_pool, device=self.device) < (4096 / N_pool * percentage_kept)
                combined_mask = axis_mask & dropout_mask # [B, N]

                # Flatten to filter efficiently
                flat_points_x = points[combined_mask].detach().cpu()
                flat_normals_x = normals[combined_mask].detach().cpu()
                # TODO: Maybe also below offset
                axis_mask = points[..., split_axis_two] > offset
                combined_mask = axis_mask & dropout_mask

                flat_points_z = points[combined_mask].detach().cpu()
                flat_normals_z = normals[combined_mask].detach().cpu()

                features_split_x, _ = self.pc_encoder(flat_points_x)
                features_split_z, _ = self.pc_encoder(flat_points_z)

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
            if save_pc_imgs:
                pc_renderer = self.get_renderings_verts_from_file_pc(pc_path)
                pc_imgs = pc_renderer.render_mvdream_views(flat_points_x, camera=self.camera)
                save_training_views_grid(
                    imgs=pc_imgs,
                    out_path=os.path.join(debug_dir, f"{name}_pc_split_x.png"),
                )
                pc_imgs = pc_renderer.render_mvdream_views(flat_points_z, camera=self.camera)
                save_training_views_grid(
                    imgs=pc_imgs,
                    out_path=os.path.join(debug_dir, f"{name}_pc_split_z.png"),
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
                "pc_feat_x_split": features_split_x.squeeze(0).detach().cpu(),
                "pc_feat_z_split": features_split_z.squeeze(0).detach().cpu(),
                "z": z.to("cpu", non_blocking=False),
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

        z = item["z"].to(self.device).contiguous()
        pc_feat = item["pc_feat"].to(self.device).contiguous()
        pc_feat_mask = item["pc_feat_mask"].to(self.device).contiguous()

        #points = item["training_points"].to(self.device).contiguous()
        #normals = item["training_normals"].to(self.device).contiguous()

        c_text = item["c_text"].to(self.device).contiguous()

        B, V, C_lat, H_lat, W_lat = z.shape

        # Flatten B and V for the UNet: (B*V, ...)
        camera_flat = self.camera.unsqueeze(0).repeat_interleave(B, dim=0).view(B * V, 16)
        z_flat = z.view(B * V, C_lat, H_lat, W_lat)
        """
        camera_flat = cameras.view(B * V, 16)
        if not torch.equal(check_cam, camera_flat):
            raise ValueError("Batched Camera not reproduced")
        else:
            print("Camera check passed")
        """
        #camera_flat = self.camera.repeat_interleave(B, dim=0).view(B * V, 16)
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

        loss = self.compiled_loss(z_noisy, z_flat, t, c_text_flat, pc_feat, pc_feat_mask, camera_flat, V, noise)

        return float(loss.item())

    @torch.no_grad()
    def validation_step(
            self,
            batch: dict,
            steps: int = 30,
            cfg_scale: float = 7.5,
    ) -> dict:
        """
        Run a full inference pass on validation samples and compare against
        the cached ground-truth images decoded from the VAE.

        Args:
            batch:   dict produced by build_cache()
            num_samples: if set, evaluate only this many randomly chosen samples
            steps:       number of ODE/DDIM steps for generation
            cfg_scale:   classifier-free guidance scale
            seed:        RNG seed for reproducible noise

        Returns:
            dict with keys "mse" ("psnr" (mean over samples))
        """
        self.model.eval()
        self.projector.eval()

        total_mse = 0.0
        total_l1 = 0.0

        z = batch["z"].to(self.device).contiguous()
        pc_feat = batch["pc_feat"].to(self.device).contiguous()
        points = batch["training_points"].to(self.device).contiguous()
        normals = batch["training_normals"].to(self.device).contiguous()
        c_text = batch["c_text"].to(self.device).contiguous()


        B, V, C_lat, H_lat, W_lat = z.shape
        cameras_flat = self.camera.unsqueeze(0).repeat_interleave(B, dim=0).view(B * V, 16)

        # --- decode ground-truth latents to pixel space ---
        z_gt_flat = z.view(B * V, C_lat, H_lat, W_lat)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gt_pixels = self.model.decode_first_stage(z_gt_flat)   # (V, 3, H, W) in [-1,1]

        gt_pixels = torch.clamp((gt_pixels + 1.0) / 2.0, 0.0, 1.0)
        gt_pixels = (gt_pixels * 255.0).permute(0, 2, 3, 1).view(B,V,self.H, self.W,-1).to(torch.float32)



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
        lengths = combined_mask.sum(dim=1)
        batch_idx = torch.repeat_interleave(torch.arange(B_pts, device=lengths.device), lengths)

        pc = {
            "points": flat_points,
            "normals": flat_normals,
            "batch_idx": batch_idx
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            #utonia_features = self.pc_encoder(coords=pc["points"], normals=pc["normals"], batch_lengths=pc["batch_lengths"])
            utonia_features, batch_index = self.pc_encoder(coords=pc["points"], batch_idx=pc["batch_idx"])
            cameras = cameras_flat.view(B, V, 16)
            pc_tokens, pc_latent = self.projector_fwd(utonia_features, cameras, batch_index)

        # --- generate images from the point cloud ---


        gen_pixels, samples = self.sample_multiview(
            pc_tokens=pc_tokens,
            pc_latent=pc_latent,
            use_pointcloud=True,
            cameras_flat=cameras_flat,
            c_text=c_text,
            num_views=V,
            H=self.H,
            W=self.W,
            steps=steps,
            scale=cfg_scale,
        )  # (V, H, W, 3) uint8


        # --- pixel-space metrics ---
        mse = F.mse_loss(z, samples)
        l1 = F.l1_loss(gen_pixels, gt_pixels).item()
        total_mse += mse
        total_l1 += l1

        return {"mse": total_mse, "l1": total_l1}


    @torch.no_grad()
    def sample_multiview(
            self,
            pc_tokens: torch.Tensor,
            pc_latent: torch.Tensor,
            c_text: torch.Tensor,
            cameras_flat: torch.Tensor,
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

        B = c_text.shape[0]
        latent_shape = [4, H // 8, W // 8]
        if use_pointcloud:
            cond_context = torch.cat([pc_tokens], dim=1).to(self.device)         # [V,K,C]

            uc_pc_tokens = torch.zeros_like(pc_tokens, device=self.device)
            uc_context = torch.cat([uc_pc_tokens], dim=1).to(self.device)                    # [V,L+K,C]

        else:
            uc_text = self.model.get_learned_conditioning([""] * num_views).to(self.device)     # [V,L,C]
            uc_text = uc_text.repeat_interleave(B, dim=0)
            cond_context = c_text                                                  # [V,L,C]
            uc_context = uc_text                                                   # [V,L,C]

        cond = {
            "context": cond_context,
            "camera": cameras_flat,
            "num_frames": num_views,
        }

        uc = {
            "context": uc_context,
            "camera": cameras_flat,
            "num_frames": num_views,
        }
        if self.flow_matching:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                args = {
                    "num_steps" : steps,
                    "cfg_scale" : scale,
                    "cond" : cond,
                    "uc_cond" : uc,
                    "progress" : False
                }
                if self.start_from_noise:
                    x_source = torch.randn((B * num_views, 4, H // 8, W // 8), device=self.device, dtype=torch.bfloat16)
                x_source = x_source.to(self.device)
                samples = self.sampler.generate(x=x_source, sample_kwargs=args)
        else:
            with torch.amp.autocast("cuda", torch.bfloat16):
                samples, _ = self.sampler.sample(
                    S=steps,
                    conditioning=cond,
                    batch_size=B,
                    shape=latent_shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=0.0,
                    x_T=None,
                )
        x = self.model.decode_first_stage(samples)
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        x = (x * 255.0).permute(0, 2, 3, 1).view(B,self.num_views,self.H, self.W,-1)
        samples = samples.view(B, num_views, 4, 32, 32)
        return x.to(torch.float32), samples