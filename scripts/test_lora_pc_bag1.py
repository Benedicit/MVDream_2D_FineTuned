import os
import sys
from pathlib import Path
import torch

from tester import Tester3D
from mvdream.camera_utils import get_camera
from test_pointnet_encoder import get_point_cloud_name, get_point_cloud_name_reg

working_dir = os.path.dirname(os.path.realpath(__file__))
print(working_dir)
SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = working_dir + "/../debug"
MESH_DIR = working_dir + "/../debug_3D"
print(SNAP_DIR)

def test_samples(train_samples):
    base_path = "/home/bweiss/Benedikt/ShapeDream/data/dataset_masked/"
    save_path = "/home/bweiss/Benedikt/ShapeDream/mvdream_2D/debug/"
    # load from specific checkpoint
    tester = Tester3D(ckpt_path="checkpoints/mvdream_lora_pc_256_classes_chair_interleaved_v2.pt")
    # first need to load model such that we can sample properly

    os.makedirs("samples", exist_ok=True)

    tester.load_model_for_pc(pointcloud_path=base_path + train_samples[0], lora_rank=32)

    for sample in train_samples:
        
        prompt = get_point_cloud_name_reg(sample)
        full_name = get_point_cloud_name_reg(sample, with_number=True)

        imgs_pc = tester.sample_multiview(
            pointcloud_path=base_path + sample,
            prompt=f"a {prompt}",
            use_pointcloud=True,
        )
        tester.save_view_grid(imgs_pc,  f"samples/{full_name}.png")

        obj_name = save_path + full_name
        #os.makedirs(obj_name, exist_ok=True)
        
        tester.save_4_views(imgs_pc, out_dir=obj_name)
        
        tester.views_to_3D(obj_name)



def overfit_bag():
    pointcloud_path = "/home/bweiss/Benedikt/ShapeDream/data/dataset_masked/shoe1.ply"
    tester = Tester3D()

    os.makedirs("samples", exist_ok=True)
    # get pointcloud_name to store it properly
    obj_name = get_point_cloud_name(pointcloud_path)

    # create 3D from finetuned mvdream -> 
    os.makedirs(obj_name, exist_ok=True)
    obj_name = "shoe0"
    obj_name += "_pc"
    tester.views_to_3D(obj_name)



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    train_samples = [
    #"shapenet_chair1.ply",
    #"shapenet_chair128.ply",
    "shapenet_chair262.ply",
    "shapenet_chair263.ply",
    #"shapenet_chair264.ply",
    "shapenet_chair265.ply",
    #"shapenet_chair266.ply",
    #"shapenet_chair267.ply",
    ]
    test_samples(train_samples)
    #overfit_bag()