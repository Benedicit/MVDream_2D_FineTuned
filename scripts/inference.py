import os
import sys
from pathlib import Path
import torch
from lightning import seed_everything

from tester import Tester3D, make_gt_of_sample_list
from pointnet_encoder import get_point_cloud_name, get_point_cloud_name_reg
from util import get_mesh_from_pc

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

working_dir = os.path.dirname(os.path.abspath(__file__))
print(working_dir)
SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = working_dir + "/../debug"
MESH_DIR = working_dir + "/../debug_3D"

base_path_pcn = f"/home/stud/weisb/ShapeDream/data/.pcn/ShapeNetCompletion/val/partial/02691156"
#base_path_pcn = f"/home/stud/weisb/ShapeDream/data/input_pc"
base_path = f"/home/stud/weisb/ShapeDream/data/.shapenet"
save_path = f"{working_dir}/debug/"

def test_samples(tester: Tester3D, test_samples, pcn=True,):

    os.makedirs("samples", exist_ok=True)

    #make_gt_of_sample_list(tester, test_samples, save_grid=False, save_4_views=True, generate_3D=True, pcn=pcn)
    for sample in test_samples:

        if pcn:
            prompt = "object"


            root_dir = str(Path(__file__).parent.parent.parent.absolute())
            pc_path = sample
            rel        = sample.relative_to(Path(root_dir) / "data/.pcn/ShapeNetCompletion/val/partial")  # synset_id/obj_id/00.pcd
            synset_id  = rel.parts[0]
            obj_id     = rel.parts[1]
            full_name = obj_id
            """
            pc_path = sample
            full_name = sample.stem
            """
        else:

            prompt = get_point_cloud_name_reg(sample)
            full_name = get_point_cloud_name_reg(sample, with_number=True)
            pc_path = get_mesh_from_pc(sample)

        imgs_pc = tester.sample_multiview(
            steps=30,
            pointcloud_path=pc_path,
            prompt=f"a {prompt}",
            use_pointcloud=True,
            save_pc_renders=False,
            use_text=False,
            pcn=pcn,
        )
        #tester.save_view_grid(imgs_pc,  f"samples/{full_name}_samples.png")

        obj_path = save_path + full_name
        #os.makedirs(obj_name, exist_ok=True)
        
        tester.save_4_views(imgs_pc, out_dir=obj_path)
        try:
            tester.images_to_3D(obj_path, refine_texture=False)
        except Exception:
            print("3D shape couldn't be saved")
            continue

if __name__ == "__main__":
    working_dir = str(Path(__file__).parent.parent.parent.absolute()) + "/mvdream_2D/scripts"

    seed_everything(42)
    tester = Tester3D(ckpt_path=f"{working_dir}/checkpoints/shapedream_64_lora.ckpt",
                      lora_rank=64,
                      alpha=16.0,
                      flow_matching=True,
                      start_from_noise=True,
                      )

    torch.set_float32_matmul_precision('high')
    '''
    train_samples = []
    classes = ["chair"]
    for cl in classes:
        for i in range(4250, 4500):
            train_samples.append(f"shapenet_chair{i}")
    '''
    #train_samples = list(Path(base_path_pcn).glob("**/*.ply"))
    train_samples = list(Path(base_path_pcn).glob("**/*.pcd"))
    train_samples = sorted(train_samples)
    test_samples(tester, train_samples, pcn=True)
    #make_gt_of_sample_list(tester, train_samples, generate_3D=True)