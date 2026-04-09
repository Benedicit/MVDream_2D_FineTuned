import os
import random
import sys
from pathlib import Path

import pytorch3d
import torch
from PIL import Image, ImageOps
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from rembg import new_session, remove

from flow_matching import FlowMatching
from lora import add_lora_to_cross_att_only, add_lora_to_all_layers
from mvdream.camera_utils import get_camera, create_camera_to_world_matrix
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from pc_encoder import PointCloudTransformer
from pointnet_encoder import get_pointnet_features, PointFeatProjector, read_from_plyfile
from yanx_pointnet2_encoder import YanxPointNet2Encoder

working_dir = str(Path(__file__).parent.parent.parent.absolute())
print(working_dir)
SNAP_DIR = f"{working_dir}/snap_gtr"
OUTPUT_DIR = working_dir + "/mvdream_2D/debug"
MESH_DIR = working_dir + "/mvdream_2D/debug_3D"
SHAPEDREAM_DIR = f"{working_dir}"
print(SNAP_DIR)
sys.path.insert(0, SHAPEDREAM_DIR)
if SNAP_DIR not in sys.path:
    sys.path.insert(0, str(SNAP_DIR))

from snap_gtr.scripts import inference

from view_renderer import PointRenderer
import math
import numpy as np
from pathlib import Path
from transformers import AutoModelForImageSegmentation

from pc_encoder import PointCloudEncoder

import pandas as pd
script_dir = os.path.dirname(os.path.abspath(__file__))

gso_csv = f"{script_dir}/../../data/gso_label_to_mesh.csv"
shapenet_csv = f"{script_dir}/../../data/shapenet_label_to_mesh.csv"
mapping_gso = pd.read_csv(gso_csv)
mapping_shapenet = pd.read_csv(shapenet_csv)

def get_mesh_from_pc(pointcloud_name="bag1.ply"):
    if pointcloud_name.startswith("shapenet"):
        return mapping_shapenet.loc[mapping_shapenet["label"] == pointcloud_name, "filename"].iloc[0]
    return mapping_gso.loc[mapping_gso["label"] == pointcloud_name, "filename"].iloc[0]

class Tester3D:
    def __init__(self, ckpt_path = "checkpoints/mvdream_lora_pc_shoes_multiview.pt", ELEV_DEG=15.0, AZIM_START=0.0, AZIM_SPAN=360.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.ckpt_path = ckpt_path
        self.ELEV_DEG = ELEV_DEG
        self.AZIM_START = AZIM_START
        self.AZIM_SPAN = AZIM_SPAN
        self.rembg_session = new_session("u2net")

        self.pointnet = YanxPointNet2Encoder(
            normal_channel=False,
            out_dim=256,
            device=self.device,
        )
        self.pytorch3d_io = pytorch3d.io.IO()

        self.birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True).to(self.device)
        self.birefnet.eval()
        #self.birefnet.half()
    def load_model_for_pc(self, pointcloud_path, model="sd-v2.1-base-4view", flow_matching=False, lora_rank=32, alpha=8.0):
        
        self.model = build_model(model)
        self.model.to(self.device)
        self.model.device = self.device
        self.unet = self.model.model.diffusion_model
        self.unet.to(self.device)
        if flow_matching:
            add_lora_to_all_layers(self.unet, r=lora_rank, alpha=alpha)
        else:
            add_lora_to_cross_att_only(self.unet, r=lora_rank, alpha=alpha)

        dummy_c = self.model.get_learned_conditioning(["dummy"]).to(self.device)
        context_dim = dummy_c.shape[-1]


        with torch.no_grad():
            pc_feat_dummy = get_pointnet_features(self.pointnet, pointcloud_path=pointcloud_path, device=self.device)
        pc_feat_dim = pc_feat_dummy.shape[-1]

        #self.depth_map_encoder = DepthViTTokenEncoder(token_dim=context_dim, tokens_per_view=1, num_views=4).to(
            #self.device)
        projector = PointFeatProjector(
        in_dim=pc_feat_dim,
        context_dim=context_dim,
        num_tokens=4,
        ).to(self.device)
        self.projector = PointCloudTransformer()
        self.encoder = PointCloudEncoder()

        # load module
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.unet.load_state_dict(ckpt["unet"], strict=False)
        self.projector.load_state_dict(ckpt["projector"], strict=True)

        self.model.device = self.device
        self.model.eval()
        self.projector.eval()


        self.flow_matching = flow_matching

        if self.flow_matching:
            self.sampler = FlowMatching(self.device, self.model)
            self.sampler.eval()
        else:
            self.sampler = DDIMSampler(self.model)


    def get_renderings_verts_from_file_pc(self, pointcloud_path=None):
        """
        pointcloud_path: Path to pointcloud
        """
        # Get full pointcloud to train against
        points_obj = read_from_plyfile(pointcloud_path)
        verts = torch.tensor(points_obj, dtype=torch.float32, device=self.device)[:, :3]

        renderer = PointRenderer(device=self.device, image_size=256, radius=0.015)

        return renderer, verts

    @torch.no_grad()
    def sample_multiview(
        self, 
        pointcloud_path: str,
        prompt: str = "an object",
        use_pointcloud: bool = True,
        num_views: int = 4,
        H: int = 256,
        W: int = 256,
        steps: int = 100,
        scale: float = 7.5,
        seed: int = 42,
        start_from_noise=True,

    ):
        """
        Sample num_views images from MVDream with or without PointNet++ conditioning.
        Returns [V, H, W, 3] uint8 numpy.
        """

        latent_shape = [4, H // 8, W // 8]
        batch_size = num_views

        c_text = self.model.get_learned_conditioning([prompt] * num_views).to(self.device)   # [V,L,C]
        uc_text = self.model.get_learned_conditioning([""] * num_views).to(self.device)     # [V,L,C]

        pc_file = Path(pointcloud_path).name
        mesh_path = get_mesh_from_pc(pc_file)
        mesh = load_objs_as_meshes([mesh_path], device=self.device)
        points, normals = sample_points_from_meshes(mesh, num_samples=8124, return_normals=True)

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
        print(flat_points.shape, flat_normals.shape, batch_idx.shape)

        pc = {
            "points": flat_points,
            "normals": flat_normals,
            "batch_idx": batch_idx
        }
        self.encoder.B = B_pts

        utonia_features, batch_idx = self.encoder(coords=pc["points"], batch_idx=pc["batch_idx"])


        self.camera = get_camera(
            num_frames=num_views,
            elevation=self.ELEV_DEG,
            azimuth_start=self.AZIM_START,
            azimuth_span=self.AZIM_SPAN,
            blender_coord=False,
        ).to(self.device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            cameras = self.camera
            pc_tokens, pc_latent = self.projector(utonia_features, cameras, batch_idx)  # [V,K,C]
        if use_pointcloud:

            #cond_context = torch.cat([c_text, pc_tokens], dim=1)                   # [V,L+K,C]
            cond_context = torch.cat([pc_tokens], dim=1).to(self.device)                   # [V,L+K,C]

            uc_pc_tokens = torch.zeros_like(pc_tokens, device=self.device)
            uc_context = torch.cat([uc_pc_tokens], dim=1).to(self.device)                    # [V,L+K,C]
            #uc_context = torch.cat([uc_pc_tokens], dim=1)                    # [V,L+K,C]
        else:
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
            pc_renderer, verts_pc = self.get_renderings_verts_from_file_pc(pointcloud_path)
            if start_from_noise:
                pc_imgs = pc_renderer.render_mvdream_views(verts_pc, camera=self.camera).to(self.device)
                x_source = self.model.encode_first_stage(pc_imgs)
                if hasattr(self.model, "get_first_stage_encoding"):
                    x_source = self.model.get_first_stage_encoding(x_source)
                x_source = torch.randn_like(x_source)
            else:
                noise = torch.randn_like(pc_latent) * 0.15
                x_source = pc_latent + noise

            args = {
                "num_steps" : steps,
                "cfg_scale" : scale,
                "cond" : cond,
                "uc_cond" : uc,
            }
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
        x = (x * 255.0).permute(0, 2, 3, 1).cpu().numpy()

        return x.astype(np.uint8)

    def save_view_grid(self, images_np, out_path: str, pad: int = 8):
        """
        images_np: [V, H, W, 3] uint8
        Saves a 1xV grid.
        """
        V, H, W, C = images_np.shape
        canvas_h = H + 2 * pad
        canvas_w = V * W + (V + 1) * pad

        canvas = np.zeros((canvas_h, canvas_w, C), dtype=np.uint8)
        y = pad
        for i in range(V):
            x = pad + i * (W + pad)
            canvas[y:y + H, x:x + W, :] = images_np[i]

        Image.fromarray(canvas).save(out_path)
        print("Saved", out_path)

    def views_to_3D(self, object_path):
        #output_dir = OUTPUT_DIR + "/" + object_name
        out_dir = Path(object_path)
        in_dir  = Path(object_path)
        inference.main(
            args=[
                "--ckpt_path", str(Path(SNAP_DIR) / "ckpts/full_checkpoint.pth"),
                "--in_dir",    str(in_dir),
                "--out_dir",   str(out_dir),
            ],
            standalone_mode=False,
        )
        gifs = [f for f in in_dir.iterdir() if f.suffix.lower() == ".gif"]
        for gif in gifs:
            Path(gif).unlink()

    def remove_bg_with_rembg(self, rgb_u8: np.ndarray, border_size=32) -> np.ndarray:
        img = Image.fromarray(rgb_u8, "RGB")
        padded = ImageOps.expand(img, border=border_size, fill="white")
        fg = remove(padded, session=self.rembg_session)  # returns RGBA
        w, h = fg.size
        fg = fg.crop((border_size, border_size, w - border_size, h - border_size))
        return np.array(fg, dtype=np.uint8)  # (H,W,4)
    

    def alpha_from_corner_key(self, rgb_u8: np.ndarray, pad=16, thresh=0.10) -> np.ndarray:
        """
        Estimate background color from corners and threshold RGB distance.
        thresh ~ 0.06..0.15 usually works for white backgrounds.
        Returns uint8 alpha in {0,255}.
        """
        rgb = rgb_u8.astype(np.float32) / 255.0
        H, W, _ = rgb.shape

        corners = np.concatenate([
            rgb[:pad, :pad].reshape(-1, 3),
            rgb[:pad, -pad:].reshape(-1, 3),
            rgb[-pad:, :pad].reshape(-1, 3),
            rgb[-pad:, -pad:].reshape(-1, 3),
        ], axis=0)
        bg = np.median(corners, axis=0)

        dist = np.linalg.norm(rgb - bg[None, None, :], axis=-1)
        a = (dist > thresh).astype(np.uint8) * 255
        return a

    @torch.no_grad()
    def remove_bg_with_birefnet(self, rgb_u8: np.ndarray) -> np.ndarray:
        # BiRefNet works best on 1024x1024 inputs
        H, W, _ = rgb_u8.shape
        img_input = Image.fromarray(rgb_u8)
        img_resized = img_input.resize((1024, 1024), Image.BILINEAR)

        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        pred = self.birefnet(img_tensor)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        # BiRefNet returns logits, use sigmoid to get mask
        pred = torch.sigmoid(pred)
        mask = pred[0, 0].cpu().numpy()

        # Resize mask back to original size
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        alpha_u8 = np.array(mask_img)

        # Refine mask: anything very low becomes 0, anything very high becomes 255
        # This helps preventing the "semi-transparent" look that deletes object parts
        alpha_u8[alpha_u8 < 15] = 0
        alpha_u8[alpha_u8 > 240] = 255

        rgba = np.dstack([rgb_u8, alpha_u8]).astype(np.uint8)
        return rgba

    def save_4_views(self, images_np, out_dir: str, dist=2.5, fov_deg=50.0):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        V, H, W, C = images_np.shape
        assert C == 3

        for i in range(V):
            rgb = images_np[i].astype(np.uint8)
            
            rgba = self.remove_bg_with_birefnet(rgb)
            Image.fromarray(rgba, "RGBA").save(out_dir / f"rgb_{i:03d}.png")


        self.write_snapgtr_cameras_from_angles(
            out_dir=str(out_dir),
            fov_deg=fov_deg,
            H=H, W=W,
            elev_deg=self.ELEV_DEG,
            radius=dist,
        )

    def _fov_to_intrinsic(self, fov_degree, width, height):
        fov_radian = math.radians(fov_degree)
        f = width / (2.0 * math.tan(fov_radian / 2.0))
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([[f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1]], dtype=np.float32)
        return K

    def _get_cam_pose(self, theta_deg, phi_deg, radius):
        theta = np.radians(theta_deg)
        phi = np.radians(phi_deg)
        y = radius * np.cos(theta)
        x = radius * np.sin(theta) * np.cos(phi)
        z = radius * np.sin(theta) * np.sin(phi)
        return np.array([x, y, z], dtype=np.float32)

    def _get_c2w_opencv(self, eye, center, up=np.array([0.0, -1.0, 0.0], dtype=np.float32)):
        forward = (center - eye).astype(np.float32)
        forward /= (np.linalg.norm(forward) + 1e-8)

        right = np.cross(up, forward)
        right /= (np.linalg.norm(right) + 1e-8)

        new_up = np.cross(forward, right)
        new_up /= (np.linalg.norm(new_up) + 1e-8)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.column_stack((right, new_up, forward))
        c2w[:3, 3] = eye
        return c2w

    def write_snapgtr_cameras_from_angles(self, out_dir: str, fov_deg: float, H: int, W: int,
                                          elev_deg: float, radius: float,
                                          azims_deg=[0.0,90.0,180.0,270.0]):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        K = self._fov_to_intrinsic(fov_deg, W, H)

        for i in range(len(azims_deg)):
            az = azims_deg[i]
            c2w = create_camera_to_world_matrix(elev_deg, az)

            # Scale translation by radius
            c2w[:3, 3] *= radius
            eye = c2w[:3, 3]

            c2w_cv = self._get_c2w_opencv(eye, np.array([0,0,0]))

            # Get w2c for SnapGTR
            w2c = np.linalg.inv(c2w_cv)

            p = out_dir / f"cam_{i:03d}.txt"
            with p.open("w") as f:
                f.write("extrinsic\n")
                for r in range(4):
                    f.write(" ".join(f"{w2c[r, c]:.6f}" for c in range(4)) + "\n")
                f.write("\n")
                f.write("intrinsic fx, fy, cx, cy, height, width \n")
                f.write(f"{K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f} {H} {W}\n")

