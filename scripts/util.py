import os
from PIL import Image as PilImage
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import open3d as o3d

script_dir = os.path.dirname(os.path.abspath(__file__))

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

gso_csv = f"{script_dir}/../../data/gso_label_to_mesh.csv"
shapenet_csv = f"{script_dir}/../../data/shapenet_label_to_mesh.csv"
mapping_shapenet = pd.read_csv(shapenet_csv) if os.path.exists(shapenet_csv) else None

def get_mesh_from_pc(pointcloud_name):
    return mapping_shapenet.loc[mapping_shapenet["pc_id"] == pointcloud_name, "filename"].iloc[0]

def count_label_entries(label):
    return len(mapping_shapenet[mapping_shapenet["label"] == label])

def load_pcd_to_tensor(pcd_path: str | Path) -> torch.Tensor:
    """Load a single .pcd file into a (N, 3) float32 tensor."""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points, dtype=np.float32)  # (N, 3)
    return torch.from_numpy(points)