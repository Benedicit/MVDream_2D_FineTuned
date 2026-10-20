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

base_path_pcn = f"/home/stud/weisb/ShapeDream/data/.pcn/ShapeNetCompletion/test/partial/"
#base_path_custom_input = f"/home/stud/weisb/ShapeDream/data/input_pc"
base_path_custom_input = f"/home/stud/weisb/ShapeDream/data/input_pc/with_5percent_noise"
base_path = f"/home/stud/weisb/ShapeDream/data/.shapenet"

def test_samples(tester: Tester3D, test_samples, pcn=True, save_path = f"{working_dir}/debug/", view_grid_path = None,
                 gt_path = "ground_truth/"):

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(view_grid_path, exist_ok=True)
    os.makedirs(f"{working_dir}/{gt_path}", exist_ok=True)

    make_gt_of_sample_list(tester, test_samples, save_grid=False, save_4_views=True, generate_3D=True, pcn=pcn, debug_dir=gt_path)
    for sample in test_samples:

        if pcn:
            prompt = "object"


            root_dir = str(Path(__file__).parent.parent.parent.absolute())
            pc_path = sample
            rel        = sample.relative_to(Path(root_dir) / "data/.pcn/ShapeNetCompletion/test/partial")  # synset_id/obj_id/00.pcd
            rel = sample.relative_to(Path(base_path_pcn))
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
        if view_grid_path:
            tester.save_view_grid(imgs_pc,  f"{view_grid_path}/{full_name}_grid.png")

        obj_path = save_path + full_name
        #os.makedirs(obj_name, exist_ok=True)
        
        tester.save_4_views(imgs_pc, out_dir=obj_path)
        try:
            tester.images_to_3D(obj_path, refine_texture=False)
        except Exception as e:
            print("3D shape couldn't be saved")
            print(e)
            continue

if __name__ == "__main__":
    working_dir = str(Path(__file__).parent.parent.parent.absolute()) + "/mvdream_2D/scripts"

    seed_everything(42)
    

    torch.set_float32_matmul_precision('high')
    lora_rank = 64

    tester = Tester3D(ckpt_path=f"{working_dir}/checkpoints/shapedream_{lora_rank}_lora.ckpt",
                      lora_rank=lora_rank,
                      alpha=lora_rank * 0.25,
                      flow_matching=True,
                      start_from_noise=True,
                      )
    '''
    classes = [
        "bottle",
        "bathtub",
        "laptop",
        "jar",
        "bed",
        "flowerpot"
    ]

    classes = [
        "airplane",
        "cabinet",
        "car",
        "chair",
        "lamp",
        "sofa",
        "table",
        "watercraft"
    ]
    for c in classes:
        train_samples = list(Path(base_path_pcn + f"/{c}").glob("**/*.ply"))
        train_samples = sorted(train_samples)
        test_samples(tester, train_samples, pcn=True, save_path=f"{working_dir}/{lora_rank}_lora/birefnet/{c}/",
                     view_grid_path=f"{working_dir}/{lora_rank}_lora/birefnet/{c}_grid/",
                     gt_path=f"ground_truth_biref/{c}/")
    '''



    classes = [
        "02691156",
        "02933112",
        "02958343",
        "03001627",
        "03636649",
        "04256520",
        "04379243",
        "04530566",
    ]
    for c in classes:
        train_samples = list(Path(base_path_pcn + f"/{c}").glob("**/*.pcd"))
        train_samples = sorted(train_samples)
        test_samples(tester, train_samples, pcn=True, save_path=f"{working_dir}/{lora_rank}_lora/birefnet/{c}/",
                     view_grid_path=f"{working_dir}/{lora_rank}_lora/birefnet/{c}_grid/",
                     gt_path=f"ground_truth_biref/{c}/")