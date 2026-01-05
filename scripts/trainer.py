import math
import os
import torch
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
from test_pointnet_encoder import read_from_plyfile, pointNet, PointFeatProjector, get_point_cloud_name, get_point_cloud_reg
from view_renderer import PointRenderer, MeshRendererMVDream
from tqdm import tqdm
from ShapeCompletionLoss import ShapeCompletionLoss
from tester import Tester3D
import pandas as pd
working_dir = os.path.dirname(os.path.realpath(__file__))

SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = working_dir + "/../debug"
MESH_DIR = working_dir + "/../debug_3D"


def get_mesh_from_pc(pointcloud_name="bag1.ply"):
    gso_csv = "/home/bweiss/Benedikt/ShapeDream/data/gso_label_to_mesh.csv"
    mapping = pd.read_csv(gso_csv)
    return mapping.loc[mapping["label"] == pointcloud_name, "filename"].iloc[0]



class LoRATrainer:
    def __init__(self, device, lora_rank, lora_alpha, num_steps=200, model_name="sd-v2.1-base-4view", H=256, W=256, ELEV_DEG=15.0, DIST=2.5, num_views=4):
        
        self.num_steps = num_steps
        self.device = torch.device(device)
        #if self.device.type == "cuda":
        #    torch.cuda.set_device(self.device.index)  # makes any internal `.cuda()` land on cuda:2
        self.model = build_model(model_name=model_name)

        self.unet = self.model.model.diffusion_model
        add_lora_to_mvdream_unet(self.unet, r=lora_rank, alpha=lora_alpha)
        self.model.to(self.device)
        self.model.device = self.device
        self.loss_fn = ShapeCompletionLoss()

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
        self.dummy_pointcloud_path = "/home/bweiss/Benedikt/ShapeDream/data/dataset_masked/bag1.ply"
        with torch.no_grad():
            pc_feat_dummy = pointNet(pointcloud_path=self.dummy_pointcloud_path, device=self.device)
        pc_feat_dim = pc_feat_dummy.shape[-1]
        context_dim = dummy_c.shape[-1]

        projector = PointFeatProjector(
                in_dim=pc_feat_dim,
                context_dim=context_dim,
                num_tokens=4,
            ).to(self.device)
        
        for p in projector.parameters():
                p.requires_grad_(True)

        self.projector = projector

        # Define optimizer
        self.optimizer = torch.optim.AdamW( list(self.lora_params) + list(self.projector.parameters()), lr=1e-4,)
        
        self.H = H
        self.W = W
        self.ELEV_DEG = ELEV_DEG
        self.DIST = DIST
        self.num_views = num_views

        self.tester = Tester3D()

        torch.manual_seed(42)

    def get_renderings_verts_from_file_pc(self, pointcloud_path=None):
        """
        pointcloud_path: Path to pointcloud
        """
        # Get full pointcloud to train against
        points_obj = read_from_plyfile(pointcloud_path)
        verts = torch.tensor(points_obj.vertices, dtype=torch.float32, device=self.device)[:, :3]
        # Normalization done in rendering
        """
        verts = verts - verts.mean(0, keepdim=True)
        scale = verts.norm(dim=1).max()
        if scale > 0:
            verts = verts / scale
        """
        renderer = PointRenderer(device=self.device, image_size=self.H, radius=0.015)
        
        return renderer, verts

    def get_renderings_verts_from_file_mesh(self, mesh_path=None):
        """
        mesh_path: Path to a .ply or .obj file containing mesh data
        """
        mesh = None
        # 1. Load vertices and faces
        if mesh_path.endswith(".ply"):
            verts, faces = load_ply(mesh_path)
        else:
            # For .obj files, load_objs_as_meshes is often more robust
            mesh = load_objs_as_meshes([mesh_path], device=self.device)
            verts = mesh.verts_packed()
            faces = mesh.faces_packed()

        verts = verts.to(self.device)
        faces = faces.to(self.device)

        # Normalization done in rendering
        # 2. Normalize (centering and scaling)
        # Important: Only move verts; faces just reference the indices
        """
        verts = verts - verts.mean(0, keepdim=True)
        scale = verts.norm(dim=1).max()
        if scale > 0:
            verts = verts / scale
        """

        # 3. Initialize the Mesh Renderer
        # (Assuming you named the new class MeshRendererMVDream)
        renderer = MeshRendererMVDream(device=self.device, image_size=self.H)
        
        # Return renderer, verts, and faces (since the renderer now needs both)
        return renderer, verts, faces, mesh if mesh else None

    def train_with_point_cloud(self, pc_feat, x0 : str, save_train_img=False):
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
        debug_name = "A bag"  # get_point_cloud_reg(x0)

        # TODO: Add Depth Map conditioning
        condition_imgs = renderer.render_mvdream_views(
            verts, camera=camera
        )

        # Encode conditional images to latent space
        with torch.no_grad():
            z_cond = self.model.encode_first_stage(condition_imgs)
            if hasattr(self.model, "get_first_stage_encoding"):
                z_cond = self.model.get_first_stage_encoding(z_cond)

        if save_train_img:
            self.save_training_views_grid(imgs=condition_imgs, out_path="debug/" + debug_name + "_check_pc" + ".png")


        mesh_path = get_mesh_from_pc()

        mesh_renderer, verts_m, faces_m, mesh = self.get_renderings_verts_from_file_mesh(mesh_path)

        target_imgs = mesh_renderer.render_mvdream_views(verts_m, faces_m, camera=camera)

        if save_train_img:
            self.save_training_views_grid(imgs=target_imgs, out_path="debug/" + debug_name + "_target" + ".png")

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
            context = torch.cat([c_text, pc_tokens], dim=1)

            cond = {"context": context, "camera": camera, "num_frames": V, }

            eps_pred = self.model.apply_model(z_noisy, t, cond)

            loss = F.mse_loss(eps_pred, noise)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            pbar.set_description(f"Image Loss: [{loss.item():.6f}] Training Steps:")





    def train_single_image(self, pc_feat, x0 : str, save_train_img=False):
        renderer, verts, faces, mesh = self.get_renderings_verts_from_file_mesh(x0)
        
        V = 4
        # get camera and input it into our mesh_renderer
        camera = get_camera(
            num_frames=V,
            elevation=self.ELEV_DEG,
            azimuth_start=0.0,
            azimuth_span=360.0,     
            blender_coord=True,     
        ).to(self.device)
        debug_name = "A bag"#get_point_cloud_reg(x0)
        
        imgs = renderer.render_mvdream_views(
            verts, faces, camera=camera
        )
        
        if save_train_img:
            self.save_training_views_grid(imgs=imgs, out_path="/home/bweiss/Benedikt/ShapeDream/mvdream_2D/debug/"+debug_name+"check"+".png")
        
        with torch.no_grad():
            z = self.model.encode_first_stage(imgs)
            if hasattr(self.model, "get_first_stage_encoding"):
                z = self.model.get_first_stage_encoding(z)
        V = self.num_views   
        
        pc_feat_fixed = pc_feat.detach().to(self.device)     
        pc_feats_views = pc_feat_fixed.expand(V, -1)         
        
        V = imgs.shape[0]  
        azims = torch.tensor([0.0, 90.0, 180.0, 270.0], device=self.device)[:V]



        cam = camera.detach().cpu().numpy().reshape(V,4,4)
        centers = cam[:, :3, 3]
        print("camera centers:", centers)

        self.model.train()
        pc_prompt = "a " + get_point_cloud_name(x0)
        
        #TODO: change to converge criterion
        pbar = tqdm(range(self.num_steps))
        for _ in pbar:
            # t ~ [1, ..., T]
            t_scalar = torch.randint(low=0, high=self.model.num_timesteps, size=(1,),device=self.device,dtype=torch.long,)  
            t = t_scalar.expand(V)  
            
            # z_noisy ~ noise
            noise = torch.randn_like(z)  
            z_noisy = self.model.q_sample(z, t, noise)

            # get prior from model
            prompts = [pc_prompt] * V
            c_text = self.model.get_learned_conditioning(prompts).to(self.device)  

            # get pc_tokens to induce in conditioning
            pc_tokens = self.projector(pc_feats_views)                       
            context = torch.cat([c_text, pc_tokens], dim=1) 

            cond = {"context": context,"camera": camera,"num_frames": V,}

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
        torch.save({"unet": self.unet.state_dict(), "projector": self.projector.state_dict()}, ckpt_path)
        print("Saved", ckpt_path)

    def save_training_views_grid(self, imgs, out_path, pad=16):
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