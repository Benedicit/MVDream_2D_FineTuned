import os
import sys
from pathlib import Path
import torch
from lightning import seed_everything

from tester import Tester3D, make_gt_of_sample_list
from pointnet_encoder import get_point_cloud_name, get_point_cloud_name_reg

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

working_dir = os.path.dirname(os.path.abspath(__file__))
print(working_dir)
SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = working_dir + "/../debug"
MESH_DIR = working_dir + "/../debug_3D"

base_path = f"{working_dir}/../../data/dataset_masked/"
save_path = f"{working_dir}/debug2/"

def test_samples(tester: Tester3D, test_samples):

    os.makedirs("samples", exist_ok=True)

    #make_gt_of_sample_list(tester, test_samples, save_grid=False, save_4_views=True, generate_3D=True)

    for sample in test_samples:
        
        prompt = get_point_cloud_name_reg(sample)
        full_name = get_point_cloud_name_reg(sample, with_number=True)

        imgs_pc = tester.sample_multiview(
            steps=30,
            pointcloud_path=base_path + sample,
            prompt=f"a {prompt}",
            use_pointcloud=True,
            start_from_noise=True,
            save_pc_renders=False,
        )
        #tester.save_view_grid(imgs_pc,  f"samples/{full_name}_samples.png")

        obj_path = save_path + full_name
        #os.makedirs(obj_name, exist_ok=True)
        
        tester.save_4_views(imgs_pc, out_dir=obj_path)
        
        tester.images_to_3D(obj_path, refine_texture=False)

if __name__ == "__main__":
    working_dir = str(Path(__file__).parent.parent.parent.absolute()) + "/mvdream_2D/scripts"
    seed_everything(42)
    tester = Tester3D(ckpt_path=f"{working_dir}/checkpoints/shapedream_flowmatching_utonia_1_classes_2000_2l_dist-v1.ckpt",
                      lora_rank=64,
                      flow_matching=True,
                      )
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
        for i in range(4250, 4500):
            train_samples.append(f"shapenet_chair{i}")
            #train_samples.append(f"shapenet_chair1513.ply")
    test_samples(tester, train_samples)
    #make_gt_of_sample_list(tester, train_samples, generate_3D=True)