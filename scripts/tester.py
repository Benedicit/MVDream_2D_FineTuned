import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from mvdream.model_zoo import build_model
from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler

from lora import add_lora_to_mvdream_unet, LoRALinear
from test_pointnet_encoder import pointNet, PointFeatProjector, get_point_cloud_name

from pathlib import Path
from rembg import new_session, remove

working_dir = os.path.dirname(os.path.realpath(__file__))
print(working_dir)
SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = working_dir + "/../debug"
MESH_DIR = working_dir + "/../debug_3D"
print(SNAP_DIR)
sys.path.insert(1, SNAP_DIR+"/..")
from snap_gtr.scripts import inference, prepare_mv


import math
import numpy as np
from pathlib import Path


class Tester3D:
    def __init__(self, ckpt_path = "checkpoints/mvdream_lora_pc_bag1_multiview.pt", ELEV_DEG=15.0, AZIM_START=0.0, AZIM_SPAN=360.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.ckpt_path = ckpt_path
        self.ELEV_DEG = ELEV_DEG
        self.AZIM_START = AZIM_START
        self.AZIM_SPAN = AZIM_SPAN
        self.rembg_session = new_session("u2net")  

    def load_model_for_pc(self, pointcloud_path, model="sd-v2.1-base-4view"):
        
        self.model = build_model(model)
        self.model.to(self.device)
        self.model.device = self.device
        self.unet = self.model.model.diffusion_model
        add_lora_to_mvdream_unet(self.unet, r=32, alpha=8.0)

        dummy_c = self.model.get_learned_conditioning(["dummy"]).to(self.device)
        context_dim = dummy_c.shape[-1]

        with torch.no_grad():
            pc_feat_dummy = pointNet(pointcloud_path=pointcloud_path, device=self.device)
        pc_feat_dim = pc_feat_dummy.shape[-1]

        projector = PointFeatProjector(
        in_dim=pc_feat_dim,
        context_dim=context_dim,
        num_tokens=4,
        ).to(self.device)
        self.projector = projector
        # load module
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.unet.load_state_dict(ckpt["unet"], strict=False)
        self.projector.load_state_dict(ckpt["projector"], strict=True)

    
    @torch.no_grad()
    def sample_multiview(
        self, 
        pointcloud_path: str,
        prompt: str = "a bag",
        use_pointcloud: bool = True,
        num_views: int = 4,
        H: int = 256,
        W: int = 256,
        steps: int = 50,
        scale: float = 7.5,
        seed: int = 42,
    ):
        """
        Sample num_views images from MVDream with or without PointNet++ conditioning.
        Returns [V, H, W, 3] uint8 numpy.
        """
        self.model.to(self.device)
        self.model.device = self.device
        self.model.eval()

        torch.manual_seed(seed)

        sampler = DDIMSampler(self.model)

        latent_shape = [4, H // 8, W // 8]
        batch_size = num_views


        c_text = self.model.get_learned_conditioning([prompt] * num_views).to(self.device)   # [V,L,C]
        uc_text = self.model.get_learned_conditioning([""] * num_views).to(self.device)     # [V,L,C]


        self.camera = get_camera(
            num_frames=num_views,
            elevation=self.ELEV_DEG,
            azimuth_start=self.AZIM_START,
            azimuth_span=self.AZIM_SPAN,
            blender_coord=True,
        ).to(self.device)

        if use_pointcloud:
            pc_feat = pointNet(pointcloud_path=pointcloud_path, device=self.device)     
            pc_feats_views = pc_feat.expand(num_views, -1)                         # [V,D_pc]
            pc_tokens = self.projector(pc_feats_views)                                  # [V,K,C]

            cond_context = torch.cat([c_text, pc_tokens], dim=1)                   # [V,L+K,C]


            uc_pc_tokens = torch.zeros_like(pc_tokens)
            uc_context = torch.cat([uc_text, uc_pc_tokens], dim=1)                    # [V,L+K,C]
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

    def views_to_3D(self, object_name):
        #output_dir = OUTPUT_DIR + "/" + object_name
        out_dir = Path(MESH_DIR) / object_name
        in_dir  = Path(OUTPUT_DIR) / object_name
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

    def save_4_views(self, images_np, out_dir: str, dist=2.5, fov_deg=50.0):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        V, H, W, C = images_np.shape
        assert C == 3

        for i in range(V):
            rgb = images_np[i].astype(np.uint8)

            # 1) make alpha from corner background (no rembg)
            a = self.alpha_from_corner_key(rgb, pad=16, thresh=0.10)

            # 2) kill tiny speckles + soften edge a bit (optional but helpful)
            a_img = Image.fromarray(a, "L").filter(ImageFilter.MinFilter(3))   # erode a little
            a_img = a_img.filter(ImageFilter.GaussianBlur(1.0))               # soft edge
            a = np.array(a_img, dtype=np.uint8)

            # 3) write RGBA (SnapGTR won’t crash)
            rgba = np.concatenate([rgb, a[..., None]], axis=-1).astype(np.uint8)
            Image.fromarray(rgba, "RGBA").save(out_dir / f"rgb_{i:03d}.png")

            # debug: save alpha to inspect
            Image.fromarray(a, "L").save(out_dir / f"_alpha_{i:03d}.png")

            cam = self.camera.detach().cpu().numpy().reshape(V,4,4)
            C_bl = cam[:, :3, 3]
            C_cv = np.stack([C_bl[:, 0], C_bl[:, 2], -C_bl[:, 1]], axis=1)

            phi_world = np.degrees(np.arctan2(C_cv[:, 2], C_cv[:, 0]))
            azims_deg = (90.0 - phi_world) % 360.0

        self.write_snapgtr_cameras_from_angles(
            out_dir=str(out_dir),
            fov_deg=fov_deg,
            H=H, W=W,
            elev_deg=self.ELEV_DEG,
            azims_deg=azims_deg.tolist(),
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
                                        elev_deg: float, azims_deg, radius: float):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        K = self._fov_to_intrinsic(fov_deg, W, H)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # SnapGTR reference mapping:
        # theta_list = 90 - elevation
        # phi_list   = 90 - azimuth
        theta = 90.0 - float(elev_deg)

        for i, az in enumerate(azims_deg):
            phi = 90.0 - float(az)

            eye = self._get_cam_pose(theta, phi, radius)
            c2w = self._get_c2w_opencv(eye, center)
            w2c = np.linalg.inv(c2w)

            p = out_dir / f"cam_{i:03d}.txt"
            with p.open("w") as f:
                f.write("extrinsic\n")
                for r in range(4):
                    f.write(" ".join(f"{w2c[r, c]:.6f}" for c in range(4)) + "\n")
                f.write("\n")
                f.write("intrinsic fx, fy, cx, cy, height, width \n")
                f.write(f"{K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f} {H} {W}\n")


