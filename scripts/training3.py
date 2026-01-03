import math
import os
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image as PilImage 
import numpy as np

from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.interface import LatentDiffusionInterface
from mvdream.camera_utils import get_camera
from mvdream.model_zoo import build_model
from lora import add_lora_to_mvdream_unet, LoRALinear
from test_pointnet_encoder import read_from_plyfile, pointNet, PointFeatProjector
from view_renderer import PointRenderer
from tqdm import tqdm
from trainer import LoRATrainer

def train_all():
    """
    Train over every file in the given base_paths
    """
    # ASSUMPTION: order is the same in both Folders, may need to change
    base_path_masked = "/home/bweiss/Benedikt/ShapeDream/data/dataset_masked/"
    base_path_unmasked = "/home/bweiss/Benedikt/ShapeDream/data/.gso/Embark_Lunch_Cooler_Blue.obj"
    
    # define which GPU to use
    trainer = LoRATrainer(device='cuda', lora_rank=32, lora_alpha=8.0, num_steps=400)
    #for file in os.listdir(base_path_masked):
        
    #filename = os.fsdecode(file)

    pointcloud_path_masked = base_path_masked #+ filename
    pointcloud_path_unmasked = base_path_unmasked #+ filename

    # Get pointNet++ features for masked input
    with torch.no_grad():
        pc_feat_dummy = pointNet(pointcloud_path=pointcloud_path_masked +"bag1.ply", device=trainer.device)

    trainer.train_single_image(pc_feat_dummy, pointcloud_path_unmasked, save_train_img=True)

    #trainer.train_with_3D(pc_feat_dummy, pointcloud_path_unmasked, save_train_img=True)

    trainer.save_weights()

if __name__ == "__main__":
    #overfit_bag()
    train_all()