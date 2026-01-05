import os
import sys
from pathlib import Path

from tester import Tester3D
from mvdream.camera_utils import get_camera
from test_pointnet_encoder import get_point_cloud_name

working_dir = os.path.dirname(os.path.realpath(__file__))
print(working_dir)
SNAP_DIR = f"{working_dir}/../../snap_gtr"
OUTPUT_DIR = working_dir + "/../debug"
MESH_DIR = working_dir + "/../debug_3D"
print(SNAP_DIR)

def main():
    pointcloud_path = "/home/bweiss/Benedikt/ShapeDream/data/dataset_masked/bag1.ply"
    tester = Tester3D()
    # first need to load model such that we can sample properly
    #tester.load_model_for_pc(pointcloud_path=pointcloud_path)

    #imgs_pc = tester.sample_multiview(pointcloud_path=pointcloud_path)
    #imgs_txt = tester.sample_multiview(pointcloud_path=pointcloud_path, use_pointcloud=False)


    os.makedirs("samples", exist_ok=True)
    # get pointcloud_name to store it properly
    obj_name = get_point_cloud_name(pointcloud_path)

    # just for sanity check
    #tester.save_view_grid(imgs_pc,  "samples/bag1_with_pointnet_p3d.png")
    #tester.save_view_grid(imgs_txt, "samples/bag1_text_only_p3d.png")
    
    # create 3D from finetuned mvdream -> 
    os.makedirs(obj_name, exist_ok=True)
  
    #"""

    obj_name += "_pc"

    #tester.save_4_views(imgs_pc, str(Path(OUTPUT_DIR)/obj_name), dist=2.5, fov_deg=50)

    tester.views_to_3D(obj_name)



if __name__ == "__main__":
    main()
