import math
import os
from pathlib import Path
import torch
import random
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image as PilImage 
import numpy as np

from pytorch3d.io import load_objs_as_meshes, load_ply
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.interface import LatentDiffusionInterface
from mvdream.camera_utils import get_camera
from mvdream.model_zoo import build_model
from lora import add_lora_to_mvdream_unet, LoRALinear
from test_pointnet_encoder import read_from_plyfile, get_pointnet_features, PointFeatProjector, get_point_cloud_name, get_point_cloud_name_reg
from view_renderer import PointRenderer, MeshRendererMVDream
from tqdm import tqdm
from tester import Tester3D
import pandas as pd
from tester import Tester3D
from mvdream.ldm.models.diffusion.ddim import DDIMSampler

working_dir = os.path.dirname(os.path.abspath(__file__))

from yanx_pointnet2_encoder import YanxPointNet2Encoder

SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = f"{working_dir}/../debug"
MESH_DIR = f"{working_dir}/../debug_3D"

gso_csv = f"f{working_dir}/../../data/gso_label_to_mesh.csv"
shapenet_csv = f"{working_dir}/../../data/shapenet_label_to_mesh.csv"
mapping_gso = pd.read_csv(gso_csv)
mapping_shapenet = pd.read_csv(shapenet_csv)


def get_mesh_from_pc(pointcloud_name="bag1.ply"):
    if pointcloud_name.startswith("shapenet"):
        return mapping_shapenet.loc[mapping_shapenet["label"] == pointcloud_name, "filename"].iloc[0]
    return mapping_gso.loc[mapping_gso["label"] == pointcloud_name, "filename"].iloc[0]

def _rgb01(imgs):
    return (0.5 * (imgs + 1.0)).clamp(0.0, 1.0)

def make_mask_from_gt(gt_imgs, pad=16, thresh=0.10, blur_iters=2):
    rgb = _rgb01(gt_imgs)  # (V,3,H,W)
    V, _, H, W = rgb.shape
    p = min(pad, H // 4, W // 4)

    tl = rgb[:, :, :p, :p]
    tr = rgb[:, :, :p, W-p:W]
    bl = rgb[:, :, H-p:H, :p]
    br = rgb[:, :, H-p:H, W-p:W]
    corners = torch.cat([tl, tr, bl, br], dim=2)
    bg = corners.mean(dim=(2, 3), keepdim=True)  # (V,3,1,1)

    dist = (rgb - bg).abs().mean(dim=1, keepdim=True)  # (V,1,H,W)
    mask = (dist > thresh).float()

    for _ in range(blur_iters):
        mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)

    return mask.clamp(0, 1)

def masked_l1(pred_imgs, gt_imgs, mask, eps=1e-6):
    pred = _rgb01(pred_imgs)
    gt = _rgb01(gt_imgs)
    per_pix = (pred - gt).abs().mean(dim=1, keepdim=True)  # (V,1,H,W)
    num = (mask * per_pix).sum()
    den = mask.sum() + eps
    return num / den


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

def make_gt_of_sample_list(samples, elev_deg=15.0, debug_dir: str = "debug/test"):
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
        verts_m = mesh.verts_packed().to(device)
        faces_m = mesh.faces_packed().to(device)

        target_imgs = renderer.render_mvdream_views(verts_m, faces_m, camera=cam).contiguous()

        save_training_views_grid(
            imgs=target_imgs,
            out_path=os.path.join(debug_dir, f"{name}_target.png"),
        )
        pbar.update(1)
class LoRATrainer:
    def __init__(self, device, lora_rank, lora_alpha, num_steps=200, model_name="sd-v2.1-base-4view", H=256, W=256, ELEV_DEG=15.0, DIST=2.5, num_views=4, load_from_ckpth : bool =False, ckpt_path="checkpoints/mvdream_lora_pc_shoes_interleaved.pt"):
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.scaler = torch.amp.GradScaler('cuda')
        #if self.device.type == "cuda":
        #    torch.cuda.set_device(self.device.index)  # makes any internal `.cuda()` land on cuda:2
        self.model = build_model(model_name=model_name)

        self.unet = self.model.model.diffusion_model
        add_lora_to_mvdream_unet(self.unet, r=lora_rank, alpha=lora_alpha)


        self.model.to(self.device)

        self.model.device = self.device

        for p in self.model.parameters():
            p.requires_grad_(False)
        
        # Define lora_params
        self.lora_params = []
        for m in self.unet.modules():
            if isinstance(m, LoRALinear):
                m.base.weight.requires_grad_(False)
                if m.base.bias is not None:
                    m.base.bias.requires_grad_(False)
                m.lora_down.weight.requires_grad_(True)
                m.lora_up.weight.requires_grad_(True)
                self.lora_params.append(m.lora_down.weight)
                self.lora_params.append(m.lora_up.weight)
        
        # Define projector properly
        dummy_c = self.model.get_learned_conditioning(["dummy"]).to(self.device)
        self.dummy_pointcloud_path = f"f{working_dir}/../../data/dataset_masked/bag1.ply"

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


        # Define optimizer
        #self.optimizer = torch.optim.AdamW( list(self.lora_params) + list(self.projector.parameters()) + list(self.depth_map_encoder.proj.parameters()), lr=1e-4,)
        self.optimizer = torch.optim.AdamW( list(self.lora_params) + list(self.projector.parameters()), lr=1e-4,)

        self.H = H
        self.W = W
        self.ELEV_DEG = ELEV_DEG
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

        #self.pointnet = torch.compile(self.pointnet)
        #self.model = torch.compile(self.model, mode="reduce-overhead")
        #self.projector = torch.compile(projector)
        #self.unet = torch.compile(self.unet, mode="reduce-overhead")

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
        else:
            # For .obj files, load_objs_as_meshes is often more robust
            mesh = load_objs_as_meshes([mesh_path], device=self.device, load_textures=textures)
            verts = mesh.verts_packed()
            faces = mesh.faces_packed()

        verts = verts.to(self.device)
        faces = faces.to(self.device)

        # 3. Initialize the Mesh Renderer

        # Return renderer, verts, and faces (since the renderer now needs both)
        return verts, faces, mesh if mesh else None

    def train_with_point_cloud(self, pc_feat, x0 : str, save_train_img=False, debug_name="A bag"):
        renderer, verts = self.get_renderings_verts_from_file_pc(x0)

        V = 4
        # get camera and input it into our mesh_renderer
        camera = get_camera(
            num_frames=V,
            elevation=self.ELEV_DEG,
            azimuth_start=0.0,
            azimuth_span=360.0,
            blender_coord=True,
        ).to(self.device)
        debug_name = debug_name  # get_point_cloud_reg(x0)

        condition_imgs = renderer.render_mvdream_views(
            verts, camera=camera
        )

        # Encode conditional images to latent space
        with torch.no_grad():
            z_cond = self.model.encode_first_stage(condition_imgs)
            if hasattr(self.model, "get_first_stage_encoding"):
                z_cond = self.model.get_first_stage_encoding(z_cond)

        if save_train_img:
            save_training_views_grid(imgs=condition_imgs, out_path="debug/" + debug_name + "_check_pc" + ".png")


        mesh_path = get_mesh_from_pc(debug_name + ".ply")

        verts_m, faces_m, mesh = self.get_renderings_verts_from_file_mesh(mesh_path)

        target_imgs = self.renderer.render_mvdream_views(verts_m, faces_m, camera=camera)

        if save_train_img:
            save_training_views_grid(imgs=target_imgs, out_path="debug/" + debug_name + "_target" + ".png")

        with torch.no_grad():
            z = self.model.encode_first_stage(target_imgs)
            if hasattr(self.model, "get_first_stage_encoding"):
                z = self.model.get_first_stage_encoding(z)
        V = self.num_views



        pc_feat_fixed = pc_feat.detach().to(self.device)
        pc_feats_views = pc_feat_fixed.expand(V, -1)

        V = condition_imgs.shape[0]

        cam = camera.detach().cpu().numpy().reshape(V, 4, 4)
        centers = cam[:, :3, 3]
        print("camera centers:", centers)

        self.model.train()
        pc_prompt = "a " + get_point_cloud_name(x0)

        # TODO: change to converge criterion
        pbar = tqdm(range(self.num_steps))

        # not sure if we need all of this inside the for loop?

        for _ in pbar:
            # t ~ [1, ..., T]
            t_scalar = torch.randint(low=0, high=self.model.num_timesteps, size=(1,), device=self.device,
                                     dtype=torch.long, )
            t = t_scalar.expand(V)

            # z_noisy ~ noise
            noise = torch.randn_like(z)
            z_noisy = self.model.q_sample(z, t, noise)

            # get prior from model
            prompts = [pc_prompt] * V
            c_text = self.model.get_learned_conditioning(prompts).to(self.device)

            # get pc_tokens to induce in conditioning
            pc_tokens = self.projector(pc_feats_views)
            #depth_tokens = self.depth_map_encoder(condition_imgs)
            #depth_tokens = depth_tokens.repeat_interleave(V, dim=0)

            #context = torch.cat([c_text, pc_tokens, depth_tokens], dim=1)
            context = torch.cat([c_text, pc_tokens], dim=1)

            cond = {"context": context, "camera": camera, "num_frames": V, }


            eps_pred = self.model.apply_model(z_noisy, t, cond)
            loss = F.mse_loss(eps_pred, noise)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            pbar.set_description(f"Image Loss: [{loss.item():.6f}] Training Steps:")

    def save_weights(self, ckpt_path="checkpoints/mvdream_lora_pc_bag1_multiview.pt"):
        """
        save the weights in specified file
        self: Description
        """
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        #torch.save({"unet": self.unet.state_dict(), "projector": self.projector.state_dict(), "depth_map_encoder": self.depth_map_encoder.state_dict()}, ckpt_path)
        torch.save({"unet": self.unet.state_dict(), "projector": self.projector.state_dict()}, ckpt_path)
        print("Saved", ckpt_path)

    @torch.no_grad()
    def build_cache(
        self,
        train_samples,
        base_path_masked: str,
        save_debug_imgs: bool = False,
        debug_dir: str = "debug/cache",
        use_fixed_camera: bool = True,
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
            pc_feat = get_pointnet_features(self.pointnet, pointcloud_path=pc_path, device=device)
            if pc_feat.dim() == 1:
                pc_feat = pc_feat.unsqueeze(0)
            pc_feat = pc_feat.detach().contiguous()


            # --- Render target multi-view images ---
            mesh_path = get_mesh_from_pc(sample)
            verts_m, faces_m, _ = self.get_renderings_verts_from_file_mesh(mesh_path)
            target_imgs = self.renderer.render_mvdream_views(verts_m, faces_m, camera=cam).contiguous()

            if save_debug_imgs:
                save_training_views_grid(
                    imgs=target_imgs,
                    out_path=os.path.join(debug_dir, f"{name}_target.png"),
                )

            # --- Encode first stage (use AMP on GPU) ---
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = self.model.encode_first_stage(target_imgs)
                if hasattr(self.model, "get_first_stage_encoding"):
                    z = self.model.get_first_stage_encoding(z)
            z = z.detach().contiguous()

            # --- Optional debug rendering of the partial point cloud views ---
            if save_debug_imgs:
                pc_renderer, verts_pc = self.get_renderings_verts_from_file_pc(pc_path)
                cond_imgs = pc_renderer.render_mvdream_views(verts_pc, camera=cam)
                save_training_views_grid(
                    imgs=cond_imgs,
                    out_path=os.path.join(debug_dir, f"{name}_pc.png"),
                )

            # --- Text conditioning: compute ONCE per prompt, then repeat V ---
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
                "name": name,
                "pc_feat": pc_feat.to("cpu", non_blocking=False),
                "z": z.to("cpu", non_blocking=False),
                "camera": cam_cpu,  # shared if fixed
                "V": V,
                "target_imgs": target_imgs.detach().to("cpu", non_blocking=False),
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

        V = int(item.get("V", self.num_views))

        z = item["z"].to(self.device)
        camera = item["camera"].to(self.device, non_blocking=True)
        pc_feat = item["pc_feat"].to(self.device, non_blocking=True)
        pc_paths = item["pc_path"]
        c_text = item["c_text"].to(self.device, non_blocking=True)

        B, V, C_lat, H_lat, W_lat = z.shape

        # Flatten B and V for the UNet: (B*V, ...)
        z_flat = z.view(B * V, C_lat, H_lat, W_lat)
        camera_flat = camera.view(B * V, 16)
        c_text_flat = c_text.view(B * V, c_text.shape[-2], c_text.shape[-1])

        t_scalar = torch.randint(0, self.model.num_timesteps, (B,), device=self.device)
        t = t_scalar.repeat_interleave(V)  # (B*V,) each object in batch gets same t per view

        noise = torch.randn_like(z_flat)
        z_noisy = self.model.q_sample(z_flat, t, noise)

        #prompts = []
        #for path in pc_paths:
       #     prompts.extend(["a " + get_point_cloud_name_reg(path)] * V)
       # c_text = self.model.get_learned_conditioning(prompts).to(self.device)

        pc_feat_flat = pc_feat.repeat_interleave(V, dim=0)
        pc_tokens = self.projector(pc_feat_flat)

        with torch.amp.autocast("cuda",  dtype=torch.bfloat16):

            context = torch.cat([c_text_flat, pc_tokens], dim=1)

            #context = torch.cat([c_text, pc_tokens], dim=1)

            cond = {"context": context, "camera": camera_flat, "num_frames": V}

            eps_pred = self.model.apply_model(z_noisy, t, cond)

            loss = F.mse_loss(eps_pred, noise)

        self.optimizer.zero_grad(set_to_none=True)
        # Scaled backward pass
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        #loss.backward()
        #self.optimizer.step()

        return float(loss.item())

    @torch.no_grad()
    def sample_multiview_2(
        self,
        pointcloud_path: str,
        prompt: str = "a shoe",
        use_pointcloud: bool = True,
        num_views: int = 4,
        H: int = 256,
        W: int = 256,
        steps: int = 50,
        scale: float = 7.5,
        seed: int = 42,
        return_torch: bool = False,
    ):
        self.model.to(self.device)
        self.model.device = self.device
        self.model.eval()

        torch.manual_seed(seed)
        sampler = DDIMSampler(self.model)

        latent_shape = [4, H // 8, W // 8]
        batch_size = num_views

        c_text = self.model.get_learned_conditioning([prompt] * num_views).to(self.device)
        uc_text = self.model.get_learned_conditioning([""] * num_views).to(self.device)

        camera = get_camera(
            num_frames=num_views,
            elevation=self.ELEV_DEG,
            azimuth_start=0.0,
            azimuth_span=360.0,  
            blender_coord=True,
        ).to(self.device)

        if use_pointcloud:
            pc_feat = get_pointnet_features(self.pointnet, pointcloud_path=pointcloud_path, device=self.device)
            if pc_feat.dim() == 1:
                pc_feat = pc_feat.unsqueeze(0)
            pc_feats_views = pc_feat.repeat(num_views, 1)

            pc_tokens = self.projector(pc_feats_views)

            renderer, verts = self.get_renderings_verts_from_file_pc(pointcloud_path)
            condition_imgs = renderer.render_mvdream_views(verts, camera=camera)

            #depth_tokens = self.depth_map_encoder(condition_imgs)  # expect (V,T,C)
            #depth_tokens = depth_tokens.repeat_interleave(4, dim=0)

            #cond_context = torch.cat([c_text, pc_tokens, depth_tokens], dim=1)
            cond_context = torch.cat([c_text, pc_tokens], dim=1)

            uc_pc_tokens = torch.zeros_like(pc_tokens)
            #uc_depth_tokens = torch.zeros_like(depth_tokens)
            uc_context = torch.cat([uc_text, uc_pc_tokens], dim=1)
        else:
            cond_context = c_text
            uc_context = uc_text

        cond = {"context": cond_context, "camera": camera, "num_frames": num_views}
        uc = {"context": uc_context, "camera": camera, "num_frames": num_views}

        samples, _ = sampler.sample(
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

        x = self.model.decode_first_stage(samples)  # (V,3,H,W) in [-1,1]

        if return_torch:
            return x  # torch, [-1,1]

        # old uint8 path
        x01 = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        x_np = (x01 * 255.0).permute(0, 2, 3, 1).cpu().numpy()
        return x_np.astype(np.uint8)


    @torch.no_grad()
    def compute_val_masked_img_loss(self, val_cache, steps=50, scale=7.5, seed=123, num_samples=8):
        self.model.eval()
        keys = list(val_cache.keys())
        assert len(keys) > 0
        rng = random.Random(seed)
        chosen = [rng.choice(keys) for _ in range(num_samples)]

        losses = []
        for k in chosen:
            item = val_cache[k]
            pc_path = item["pc_path"]
            gt_imgs = item["target_imgs"].to(self.device)  # (V,3,H,W) [-1,1]

            prompt = "a " + get_point_cloud_name(pc_path)

            pred_imgs = self.sample_multiview_2(
                pointcloud_path=pc_path,
                prompt=prompt,
                use_pointcloud=True,
                num_views=self.num_views,
                H=self.H,
                W=self.W,
                steps=steps,
                scale=scale,
                seed=seed,              # fixed seed for determinism
                return_torch=True,      # returns (V,3,H,W) [-1,1]
            ).to(self.device)

            mask = make_mask_from_gt(gt_imgs, pad=16, thresh=0.10, blur_iters=2)
            loss = masked_l1(pred_imgs, gt_imgs, mask)
            losses.append(loss.item())

        return float(sum(losses) / len(losses))
