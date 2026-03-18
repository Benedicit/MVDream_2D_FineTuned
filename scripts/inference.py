import os
import sys
from pathlib import Path
import torch

from tester import Tester3D
from test_pointnet_encoder import get_point_cloud_name, get_point_cloud_name_reg
from trainer import make_gt_of_sample_list

working_dir = os.path.dirname(os.path.abspath(__file__))
print(working_dir)
SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = working_dir + "/../debug"
MESH_DIR = working_dir + "/../debug_3D"
print(SNAP_DIR)

base_path = f"{working_dir}/../../data/dataset_masked/"
save_path = f"{working_dir}/debug/"


def test_samples(tester: Tester3D, test_samples):
    # load from specific checkpoint
    # first need to load model such that we can sample properly

    os.makedirs("samples", exist_ok=True)

    tester.load_model_for_pc(pointcloud_path=base_path + test_samples[0], lora_rank=32)
    #make_gt_of_sample_list(test_samples)

    for sample in test_samples:
        
        prompt = get_point_cloud_name_reg(sample)
        full_name = get_point_cloud_name_reg(sample, with_number=True)

        imgs_pc = tester.sample_multiview(
            pointcloud_path=base_path + sample,
            prompt=f"a {prompt}",
            use_pointcloud=False,
            #scale=80.0
        )
        #tester.save_view_grid(imgs_pc,  f"samples/{full_name}_samples.png")

        obj_path = save_path + full_name
        #os.makedirs(obj_name, exist_ok=True)
        
        tester.save_4_views(imgs_pc, out_dir=obj_path)
        
        tester.views_to_3D(obj_path)

if __name__ == "__main__":
    tester = Tester3D(ckpt_path="checkpoints/mvdream_lora_pc_720_classes_bench_raw.pt")
    torch.set_float32_matmul_precision('high')
    '''
    train_samples = [
    #"shapenet_chair1.ply",
    #"shapenet_chair128.ply",
    #"shapenet_chair140.ply",
    #"shapenet_chair450.ply",
    #"shapenet_chair451.ply",
    #"shapenet_chair452.ply",
    #"shapenet_chair453.ply",
    #"shapenet_chair454.ply",
    #"shapenet_lamp250.ply",
    #"shapenet_car250.ply",
    #"shapenet_table250.ply",
    #"shapenet_bench250.ply",
    ]
    train_samples = ["shapenet_chair401.ply"]#"shapenet_airplane250.ply", "shapenet_bathtub250.ply"]
    '''

    train_samples = []
    classes = ["bench", "chair", "car", "table"]
    classes = ["chair"]
    for cl in classes:
        for i in range(4200, 4450):
            train_samples.append(f"shapenet_chair{i}.ply")
            #train_samples.append(f"shapenet_chair1513.ply")
    test_samples(tester, train_samples)
    #make_gt_of_sample_list(tester, train_samples, generate_3D=True)