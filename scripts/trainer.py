import os
import sys

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from transformers import get_cosine_with_min_lr_schedule_with_warmup

working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, working_dir)

import pytorch3d
import torch
import torch.nn.functional as F
from pytorch3d.io import load_objs_as_meshes, load_ply

from lightning import LightningModule

from lora import add_lora_to_cross_att_only, add_lora_to_all_layers, LoRALinear
from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from pc_encoder import PointCloudTransformerSmall, PointCloudToLatent
from view_renderer import PointRenderer
from transformers import AutoModelForImageSegmentation

from flow_matching import FlowMatching

SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = f"{working_dir}/../debug"
MESH_DIR = f"{working_dir}/../debug_3D"


class LoRATrainer(LightningModule):
    def __init__(self,
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
                 no_compile=False,
                 lr=1e-4,
                 batch_size=16,
                 num_epochs=500):
        super().__init__()
        self.model = build_model(model_name=model_name)

        self.unet = self.model.model.diffusion_model
        self.flow_matching = flow_matching
        self.start_from_noise = start_from_noise
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        if flow_matching:
            add_lora_to_all_layers(self.unet, r=lora_rank, alpha=lora_alpha)
        else:
            add_lora_to_cross_att_only(self.unet, r=lora_rank, alpha=lora_alpha)

        #self.model.device = self.device

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


        #self.pc_encoder = PointCloudEncoder()
        self.no_compile = no_compile

        self.projector = PointCloudTransformerSmall(
            n_self_attn_layers=6,
            n_cross_attn_layers=3,
            num_tokens=4,
            latent_dim=192,
            n_heads=6,
        )
        
        for p in self.projector.parameters():
            p.requires_grad_(True)

        self.lora_param_list = list(self.lora_params)
        print(f"LoRA parameters: {sum(p.numel() for p in self.lora_param_list)/1e6:.2f}M, LoRA Layers: {lora_layer_count}")
        print(f"Projector parameters: {sum(p.numel() for p in self.projector.parameters())/1e6:.3f}M")

        self.H = H
        self.W = W
        self.ELEV_DEG = ELEV_DEG
        self.AZIM_START = AZIM_START
        self.AZIM_SPAN = AZIM_SPAN
        self.DIST = DIST
        self.num_views = num_views

        self.register_buffer("camera", get_camera(
            num_frames=num_views,
            elevation=self.ELEV_DEG,
            azimuth_start=self.AZIM_START,
            azimuth_span=self.AZIM_SPAN,
            blender_coord=False,
        ))

        self.ckpt_path = ckpt_path

        self.pytorch3d_io = pytorch3d.io.IO()

        self.birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
        for p in self.birefnet.parameters():
            p.requires_grad_(False)
        self.birefnet.eval()

        self.compiled_wrapper = self.training_wrapper
        self.projector_fwd = self.projector.forward

        if self.flow_matching:
            #add_lora_to_attention_and_conv(self.unet, r=lora_rank, alpha=lora_alpha)
            self.sampler = FlowMatching(self.model)
            if self.start_from_noise:
                self.latent_projector = None
            else:
                self.latent_projector = PointCloudToLatent(n_self_attn_layers=1)
                print(f"Latent projector parameters: {sum(p.numel() for p in self.latent_projector.parameters())/1e6:.3f}M")
                self.latent_projector_fwd = self.latent_projector.forward
            #self.compiled_wrapper = torch.compile(self.training_wrapper, mode="max-autotune", dynamic=False, disable=no_compile)
        else:
            self.sampler = DDIMSampler(self.model)
            self.latent_projector = None
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.flow_matching and not self.start_from_noise:
            optimizer = torch.optim.AdamW(
                self.lora_param_list + list(self.projector.parameters()) + list(self.latent_projector.parameters()),
                lr=self.lr,
                weight_decay=1e-2,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.lora_param_list + list(self.projector.parameters()),
                lr=self.lr,
                weight_decay=1e-2,
                )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            },
        }

    def setup(self, stage: str):
        if hasattr(self, "_model_configured"):
            return
        print("Activating compilation")
        self.projector_fwd = torch.compile(
            self.projector.forward,
            dynamic=True,
            disable=self.no_compile,
            )
        if self.flow_matching and not self.start_from_noise:
            self.latent_projector_fwd = torch.compile(
                self.latent_projector.forward,
                dynamic=True,
                disable=self.no_compile,
            )
        self.compiled_wrapper = torch.compile(
            self.training_wrapper,
            mode="max-autotune",
            dynamic=False,
            disable=self.no_compile,
        )

        self._model_configured = True

    def training_wrapper(self, x1, x0, cond, t):
        return self.sampler.training_losses(x1=x1, x0=x0, cond=cond, t=t)

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

    def training_step(self, batch, batch_idx):
        # Unpack batch (same logic as train_one_step_from_cache)
        z         = batch["z"].contiguous()
        pc_feat   = batch["pc_feat"].contiguous()
        pc_feat_mask = batch["pc_feat_mask"].contiguous()
        c_text    = batch["c_text"].contiguous()

        B, V, C_lat, H_lat, W_lat = z.shape
        camera_flat = self.camera.unsqueeze(0).repeat_interleave(B, dim=0).view(B * V, 16)
        z_flat      = z.view(B * V, C_lat, H_lat, W_lat)
        # If prompt is used. Only for benchmarking purposes as it has barely any effect
        c_text_flat = c_text.view(B * V, c_text.shape[-2], c_text.shape[-1])

        t = torch.rand((B,), device=self.device).repeat_interleave(V)
        #t = torch.rand((B * V,), device=self.device)

        cameras  = camera_flat.view(camera_flat.shape[0] // V, V, -1)
        #pc_tokens, pc_latent = self.projector_fwd(pc_feat, pc_feat_mask, cameras)
        pc_tokens = self.projector_fwd(pc_feat, pc_feat_mask, cameras)

        # Random dropout of tokens
        drop_mask = torch.rand(B*V, device=self.device) < 0.1
        drop_mask = drop_mask.view(B*V, 1, 1)
        null_tokens = self.projector.get_null_context(batch_size=B*V)

        pc_tokens = torch.where(drop_mask, null_tokens, pc_tokens)

        cond = {"context": pc_tokens, "camera": camera_flat, "num_frames": V}

        if self.flow_matching:
            if self.start_from_noise:
                loss = self.compiled_wrapper(x1=z_flat, x0=None, cond=cond, t=t)
            else:
                # Also add some noise to the features
                pc_feat = pc_feat + torch.randn_like(pc_feat) * 0.10
                pc_latent = self.latent_projector_fwd(pc_feat, pc_feat_mask, cameras)
                # Dropout of latent to enhance only the tokens
                if torch.rand(1).item() < 0.25:
                    latent = torch.randn_like(pc_latent) + 0.0 * pc_latent
                else:
                    noise_reg = torch.randn_like(pc_latent) * 0.30
                    latent = pc_latent + noise_reg
                loss = self.compiled_wrapper(x1=z_flat, x0=latent, cond=cond, t=t)
                reg_loss = torch.mean(pc_latent) ** 2 + (torch.std(pc_latent) - 1) ** 2
                loss = loss + 0.1 * reg_loss
        else:
            noise = torch.randn_like(z_flat)
            z_noisy = self.model.q_sample(z_flat, t, noise)
            eps_pred = self.model.apply_model(z_noisy, t, cond)
            loss = F.mse_loss(eps_pred, noise)


        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z        = batch["z"].contiguous()
        pc_feat  = batch["pc_feat"].contiguous()
        pc_feat_mask = batch["pc_feat_mask"].contiguous()
        c_text   = batch["c_text"].contiguous()

        B, V, C_lat, H_lat, W_lat = z.shape
        cameras_flat = self.camera.unsqueeze(0).repeat_interleave(B, dim=0).view(B * V, 16)
        cameras      = cameras_flat.view(B, V, 16)

        pc_tokens = self.projector(pc_feat, pc_feat_mask, cameras)
        if self.start_from_noise:
            pc_latent = torch.randn_like(z)
        else:
            pc_latent = self.latent_projector_fwd(pc_feat, pc_feat_mask, cameras)

        gen_pixels, samples = self.sample_multiview(
            pc_tokens=pc_tokens,
            pc_latent=pc_latent,
            use_pointcloud=True,
            cameras_flat=cameras_flat,
            c_text=c_text,
            num_views=V,
            H=self.H, W=self.W,
        )

        z_gt_flat = z.view(B * V, C_lat, H_lat, W_lat)
        gt_pixels = torch.clamp((self.model.decode_first_stage(z_gt_flat) + 1) / 2, 0, 1)
        gt_pixels = (gt_pixels * 255).permute(0, 2, 3, 1).view(B, V, self.H, self.W, -1)

        mse = F.mse_loss(z, samples)
        l1  = F.l1_loss(gen_pixels.float(), gt_pixels.float())

        image_size = (1024, 1024)

        gt_pixels_flat = gt_pixels.view(B*V, self.H, self.W, -1)
        gen_pixels_flat = gen_pixels.view(B*V, self.H, self.W, -1)

        # Permute to (N, C, H, W) and scale back to [0, 1]
        gt_pixels_bchw = gt_pixels_flat.permute(0, 3, 1, 2) / 255.0
        gen_pixels_bchw = gen_pixels_flat.permute(0, 3, 1, 2) / 255.0

        # Resize to 1024x1024
        gt_pixels_resized = F.interpolate(gt_pixels_bchw, size=image_size, mode='bilinear', align_corners=False)
        gen_pixels_resized = F.interpolate(gen_pixels_bchw, size=image_size, mode='bilinear', align_corners=False)

        # Normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        gt_pixels_input = normalize(gt_pixels_resized)
        gen_pixels_input = normalize(gen_pixels_resized)

        alpha_preds_gt = []
        alpha_preds_gen = []

        # Process through BiRefNet one by one to avoid OOM
        for i in range(gt_pixels_input.shape[0]):
            with torch.no_grad():
                pred_gt = self.birefnet(gt_pixels_input[i:i+1])[-1].sigmoid()
                pred_gen = self.birefnet(gen_pixels_input[i:i+1])[-1].sigmoid()
                alpha_preds_gt.append(pred_gt[0])
                alpha_preds_gen.append(pred_gen[0])

        alpha_pred_gt = torch.cat(alpha_preds_gt, dim=0)
        alpha_pred_gen = torch.cat(alpha_preds_gen, dim=0)
        l1_alpha = F.l1_loss(alpha_pred_gen, alpha_pred_gt)
        mse_alpha = F.mse_loss(alpha_pred_gen, alpha_pred_gt)

        self.log("val/mse", mse, prog_bar=True, sync_dist=True)
        self.log("val/l1_img", l1, prog_bar=False, sync_dist=True)
        self.log("val/l1_alpha", l1_alpha, prog_bar=True, sync_dist=True)
        self.log("val/mse_alpha", mse_alpha, prog_bar=False, sync_dist=True)


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

            #uc_pc_tokens = torch.zeros_like(pc_tokens, device=self.device)
            uc_pc_tokens = self.projector.get_null_context(batch_size=B * num_views)
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
                else:
                    x_source = pc_latent
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
        return x, samples