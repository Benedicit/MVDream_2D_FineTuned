import os
import random
import sys
from pathlib import Path
from torch_geometric.utils import to_dense_batch
from torchvision import transforms
from tqdm import tqdm

working_dir = str(Path(__file__).parent.parent.parent.absolute())

SNAP_DIR = f"{working_dir}/snap_gtr"
OUTPUT_DIR = working_dir + "/mvdream_2D/debug"
MESH_DIR = working_dir + "/mvdream_2D/debug_3D"
SHAPEDREAM_DIR = f"{working_dir}"
MVDREAM_DIR = f"{working_dir}/mvdream_2D/scripts"

sys.path.insert(0, working_dir)

if SNAP_DIR not in sys.path:
    sys.path.insert(0, str(SNAP_DIR))

import pytorch3d
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from loguru import logger
from torch import GradScaler

from flow_matching import FlowMatching
from lora import add_lora_to_cross_att_only, add_lora_to_all_layers
from mvdream.camera_utils import get_camera, create_camera_to_world_matrix
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from pointnet_encoder import get_pointnet_features, PointFeatProjector, read_from_plyfile, get_point_cloud_name_reg
from snap_gtr.builders.build_system import build_system
from snap_gtr.scripts.inference import load_eval_data
from snap_gtr.utils.io_utils import read_yaml, EasyDict
from snap_gtr.utils.render_utils import get_cameras, np_fov_to_intrinsic, invert_transform
from snap_gtr.engine.optimizers import Optimizers
from yanx_pointnet2_encoder import YanxPointNet2Encoder

from view_renderer import PointRenderer, MeshRendererMVDream
import math
import numpy as np
from pathlib import Path
from transformers import AutoModelForImageSegmentation

from pc_encoder import PointCloudEncoder, PointCloudTransformerSmall
from util import save_training_views_grid
import pandas as pd
from trainer import LoRATrainer

script_dir = os.path.dirname(os.path.abspath(__file__))

gso_csv = f"{script_dir}/../../data/gso_label_to_mesh.csv"
shapenet_csv = f"{script_dir}/../../data/shapenet_label_to_mesh.csv"
mapping_shapenet = pd.read_csv(shapenet_csv) if os.path.exists(shapenet_csv) else None

def get_mesh_from_pc(pointcloud_name):
    return mapping_shapenet.loc[mapping_shapenet["label"] == pointcloud_name, "filename"].iloc[0]




class Tester3D:
    def __init__(self,
                 ckpt_path,
                 ELEV_DEG=15.0,
                 AZIM_START=0.0,
                 AZIM_SPAN=360.0,
                 model="sd-v2.1-base-4view",
                 flow_matching= True,
                 lora_rank=32,
                 alpha=8.0
                 ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ckpt_path = ckpt_path
        self.ELEV_DEG = ELEV_DEG
        self.AZIM_START = AZIM_START
        self.AZIM_SPAN = AZIM_SPAN

        self.pointnet = YanxPointNet2Encoder(
            normal_channel=False,
            out_dim=256,
            device=self.device,
        )
        self.pytorch3d_io = pytorch3d.io.IO()

        self.birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True).to(self.device)
        self.birefnet.eval()
        self.birefnet.half()

        # SnapGTR config
        self.snap_gtr_config_path = f"{SNAP_DIR}/configs/config_texrefine.yaml"
        job_description = read_yaml(self.snap_gtr_config_path)
        self.config = job_description["jobs"][0]
        self.config = EasyDict(self.config)

        ckpt_path = f"{SNAP_DIR}/ckpts/full_checkpoint.pth"
        self.snap_state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if self.snap_state_dict is None:
            raise ValueError(f"Failed to load checkpoint from {ckpt_path}")
        self.snap_model = build_system(self.config, device=self.device, world_size=1)
        self.snap_model.load_state_dict(self.snap_state_dict['pipeline'], strict=True)
        self.snap_model.switch_eval()


        self.param_groups = self.snap_model.get_param_groups()
        self.snap_optimizers = Optimizers(self.config["optimizers"], self.param_groups)

        self.snap_grad_scaler = GradScaler(enabled=True, init_scale=2048)
        self.snap_grad_scaler.load_state_dict(self.snap_state_dict["scalers"])

        '''
        self.model = build_model(model)
        self.model.to(self.device)
        self.model.device = self.device
        self.unet = self.model.model.diffusion_model
        self.unet.to(self.device)
        if flow_matching:
            add_lora_to_all_layers(self.unet, r=lora_rank, alpha=alpha)
        else:
            add_lora_to_cross_att_only(self.unet, r=lora_rank, alpha=alpha)

        self.projector = PointCloudTransformerSmall(n_self_attn_layers=2, num_tokens=4)

        
        self.encoder = PointCloudEncoder()

        # load module
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.projector.load_state_dict(ckpt["projector"], strict=True)

        self.model.device = self.device
        self.model.eval()
        self.projector.eval()
        '''
        self._module = LoRATrainer.load_from_checkpoint(
            ckpt_path,
            map_location=self.device,
        )
        self._module.to(self.device)
        self._module.eval()

        # --- Expose the same attributes Tester3D used before ---
        self.model     = self._module.model
        self.unet      = self._module.unet
        self.projector = self._module.projector
        self.encoder   = PointCloudEncoder().to(self.device)

        self.flow_matching = flow_matching

        if self.flow_matching:
            self.sampler = FlowMatching(self.model)
            self.sampler.eval()
        else:
            self.sampler = DDIMSampler(self.model)


    def get_renderer(self, ):
        """
        pointcloud_path: Path to pointcloud
        """
        renderer = PointRenderer(device=self.device, image_size=256, radius=0.015)

        return renderer

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
        save_pc_renders=False,

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
        points, normals = sample_points_from_meshes(mesh, num_samples=8192, return_normals=True)

        # --- Vectorized Augmentation ---
        B_pts, N_pool, _ = points.shape
        split_axis = 0 if random.random() < 0.5 else 2
        offset = random.random() * 0.015
        percentage_kept = 0.75

        # Compute masks for the whole batch at once
        axis_mask = points[..., split_axis] > offset
        dropout_mask = torch.rand((B_pts, N_pool), device=self.device) < (4096 / N_pool * percentage_kept)
        combined_mask = axis_mask & dropout_mask # [B, N]

        # Flatten to filter efficiently
        flat_points = points[combined_mask]
        flat_normals = normals[combined_mask]


        pc = {
            "points": flat_points,
            "normals": flat_normals,
        }
        self.encoder.B = B_pts

        utonia_features, batch_idx = self.encoder(coords=pc["points"])
        pc_feat, mask = to_dense_batch(utonia_features, batch_idx)


        self.camera = get_camera(
            num_frames=num_views,
            elevation=self.ELEV_DEG,
            azimuth_start=self.AZIM_START,
            azimuth_span=self.AZIM_SPAN,
            blender_coord=False,
        ).to(self.device)

        if save_pc_renders:
            full_name = get_point_cloud_name_reg(pointcloud_path, with_number=True)
            pc_renderer = self.get_renderer()
            pc_imgs = pc_renderer.render_mvdream_views(flat_points, camera=self.camera)
            imgs_np = (0.5 * (pc_imgs + 1.0)).clamp(0,1)
            imgs_np = (imgs_np.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            self.save_view_grid(
                images_np=imgs_np,
                out_path=f"samples/{full_name}_pc.png",
            )

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pc_tokens, pc_latent = self.projector(pc_feat, mask, self.camera.unsqueeze(0))  # [V,K,C]
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
            if start_from_noise:
                x_source = torch.randn(latent_shape, device=self.device)
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

    def images_to_3D(self, object_path, render_mesh_res=512, render_nerf_res=1024, refine_texture=False):

        out_dir = Path(object_path)
        in_dir  = Path(object_path)

        # load data
        data_batch = load_eval_data(in_dir)


        for key, value in data_batch.items():
            if type(value) is torch.Tensor:
                data_batch[key] = value.unsqueeze(0).cuda(self.device, non_blocking=True)


        # set up cameras for mesh rendering
        num_frames, fov_deg, cam_distance = 50, 50, 3.5
        azimuth_deg = torch.from_numpy(np.linspace(0, 360, num=num_frames, endpoint=False, dtype=np.float32))
        elevation_deg = ([20] * num_frames)
        elevation_deg = torch.tensor(elevation_deg).float()

        cameras_render = get_cameras(
            azimuth_deg,
            elevation_deg,
            width=render_mesh_res,
            height=render_mesh_res,
            fov=fov_deg,
            camera_distance=cam_distance,
        )

        # set up cameras for NeRF rendering
        K = np_fov_to_intrinsic(fov_deg, render_nerf_res, render_nerf_res)
        camera_intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
        camera_intrinsics = torch.from_numpy(camera_intrinsics).to(self.device).unsqueeze(0).expand(num_frames, -1)

        trans_cv2blender = torch.eye(4)
        trans_cv2blender[1, 1] = -1
        trans_cv2blender[2, 2] = -1
        trans_blender2cv = trans_cv2blender.T
        w2blender_cam = cameras_render['w2c']
        w2cv_cam = trans_blender2cv[None] @ w2blender_cam
        camera_poses = invert_transform(w2cv_cam)

        data_batch["poses"] = camera_poses
        data_batch["intrinsics"] = camera_intrinsics
        data_batch["imgs"] = torch.zeros((num_frames, render_nerf_res, render_nerf_res, 3), dtype=torch.float32)
        data_batch["depths"] = torch.zeros((num_frames, render_nerf_res, render_nerf_res, 1), dtype=torch.float32)
        data_batch["mvps"] = (cameras_render['mvp_mtx'])
        data_batch["m2vs"] = (cameras_render['w2c'])
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        if refine_texture:
            self.snap_model.load_state_dict(self.snap_state_dict['pipeline'], strict=True)
            code, loss = self.snap_model.refine_texture(data_batch, self.snap_optimizers, self.snap_grad_scaler, iters=50, learn_code=True)
            logger.info(f"Refine texture done, Final loss: {loss}")
        else:
            code = None

        # Mesh Gen
        logger.info(f'Save to {out_dir}')
        mesh_file = f"{out_dir}/mesh.obj"
        logger.info(f"Provide code to extract mesh")
        mesh_list = self.snap_model.extract_geometry(data_batch, resolution=512, level=10, code=code)
        logger.info(f"Extract mesh")
        mesh_list[0].export(mesh_file)

    @torch.no_grad()
    def remove_bg_with_birefnet(self, rgb_u8: np.ndarray) -> Image.Image:
        # BiRefNet works best on 1024x1024 inputs
        H, W, _ = rgb_u8.shape
        img_input = Image.fromarray(rgb_u8)

        image_size = (1024, 1024)

        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transformed_input = transform_image(img_input).unsqueeze(0).to('cuda').half()
        preds = self.birefnet(transformed_input)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)

        mask = pred_pil.resize(img_input.size)
        img_input.putalpha(mask)

        return img_input

    def save_4_views(self, images_np, out_dir: str, dist=2.5, fov_deg=50.0):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        V, H, W, C = images_np.shape
        assert C == 3

        for i in range(V):
            rgb = images_np[i].astype(np.uint8)

            masked_image = self.remove_bg_with_birefnet(rgb)
            masked_image.save(out_dir / f"rgb_{i:03d}.png")


        self.write_snapgtr_cameras_from_angles(
            out_dir=str(out_dir),
            fov_deg=fov_deg,
            H=H, W=W,
            elev_deg=self.ELEV_DEG,
            radius=dist,
        )

    # Note: The following private functions are from SnapGTR
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
                tester.images_to_3D(obj_path)
        pbar.update(1)
