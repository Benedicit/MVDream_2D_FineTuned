import torch
import torch.nn as nn
import numpy as np
import os
import trimesh
import open3d as o3d
from PIL import Image
import point_cloud_utils as pcu
from plyfile import PlyData, PlyElement
from yanx_pointnet2_encoder import YanxPointNet2Encoder
import re

test_file = "../data/dataset/bag1.ply"
data_directory = "../data/dataset"

def get_point_cloud_name(file):
    # get name from file name such that we can input it into model properly
    file = os.path.basename(file)
    result = ""
    for a in file:
        if a.isdigit():
            break
        else:
            result += a
    return result

def get_point_cloud_reg(file):
    file = os.path.basename(file)
    result = re.findall(rf"[A-Za-z]+\d+", file)[0]
    return result

def read_from_data_folder(directory):

    point_cloud_names = []
    file_names = []

    dir = os.fsencode(directory)

    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        filename = directory + "/" + filename

        file_names.append(filename)

        point_cloud_name = get_point_cloud_name(filename)
        point_cloud_names.append(point_cloud_name)

    #print(point_cloud_names, file_names)
    return point_cloud_names, file_names


def read_from_plyfile(file):

    mesh = trimesh.load_scene(file, "ply")
    mesh = mesh.to_geometry()

    return mesh


def pointNet(pointcloud_path, device):
    print("pointnet device: ", device)
    pointnet = YanxPointNet2Encoder(
        #ckpt_path="/home/bweiss/Benedikt/ShapeDream/models/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth",
        #models_root="/home/bweiss/Benedikt/ShapeDream/models/Pointnet_Pointnet2_pytorch",
        normal_channel=False,
        out_dim=256,
        device=device,
    )

    points = read_from_plyfile(pointcloud_path)
    verts = torch.tensor(points.vertices, dtype=torch.float32)

    # keep XYZ only
    verts = verts[:, :3]

    # center and scale to unit sphere (common for PointNet)
    verts = verts - verts.mean(0, keepdim=True)
    scale = verts.norm(dim=1).max()
    if scale > 0:
        verts = verts / scale

    # sample a fixed number of points
    num_points = 2048
    if verts.shape[0] >= num_points:
        idx = torch.randperm(verts.shape[0])[:num_points]
        verts = verts[idx]
    else:
        # pad by repeating
        repeat = num_points - verts.shape[0]
        extra = verts[torch.randint(0, verts.shape[0], (repeat,))]
        verts = torch.cat([verts, extra], dim=0).to(device)


    mesh = verts.unsqueeze(0)

    if pointnet.device != mesh.device:
            mesh = mesh.to(device)

    with torch.no_grad():
        feats = pointnet(mesh)

    return feats

class PointFeatProjector(nn.Module):
    def __init__(self, in_dim, context_dim=768, num_tokens=4, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.num_tokens = num_tokens
        self.context_dim = context_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_tokens * context_dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, in_dim]   (here B=1)
        return: [B, num_tokens, context_dim]
        """
        B, _ = feat.shape
        out = self.mlp(feat)  # [B, num_tokens * context_dim]
        return out.view(B, self.num_tokens, self.context_dim)


def main():

    points = read_from_plyfile(test_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = YanxPointNet2Encoder(
        ckpt_path="/home/bweiss/Benedikt/ShapeDream/models/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth",
        models_root="/home/bweiss/Benedikt/ShapeDream/models/Pointnet_Pointnet2_pytorch/models",
        normal_channel=False,
        out_dim=256,
        device=device,
    )

    mesh = torch.tensor(points.vertices, dtype=torch.float32)
    mesh = mesh[:,:3].unsqueeze(0)

    with torch.no_grad():
        feats = encoder(mesh)
    feats = np.array(feats.to("cpu"))
    flat = feats.reshape(feats.shape[0], -1).astype(np.float32)
    img = flat
    min_val = img.min()
    max_val = img.max()

    img_norm = (img - min_val) / (max_val - min_val)
    img_uint8 = (img_norm * 255.0).clip(0,255).astype(np.uint8)


    im = Image.fromarray(img_uint8, mode="L")
    im.save("pointnet_features.png")

    print("Input shape:", points.shape)
    print("Feature shape:", feats.shape)
    print("Feature sample:", feats[0, :5])


if __name__ == "__main__":
    main()
