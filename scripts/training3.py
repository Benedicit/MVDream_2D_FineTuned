import math
import os
import random
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from omegaconf import OmegaConf
from PIL import Image as PilImage 
import numpy as np

from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.interface import LatentDiffusionInterface
from mvdream.camera_utils import get_camera
from mvdream.model_zoo import build_model
from lora import add_lora_to_mvdream_unet, LoRALinear
from test_pointnet_encoder import read_from_plyfile, get_pointnet_features, PointFeatProjector, get_point_cloud_name_reg
from view_renderer import PointRenderer
from tqdm import tqdm
from trainer import LoRATrainer, get_mesh_from_pc
from tester import Tester3D

class CacheDataset(Dataset):
    def __init__(self, cache, samples):
        self.cache = cache
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = self.cache[s]
        # Return tensors directly for DataLoader collation
        return {
            "z": item["z"],
            "camera": item["camera"],
            "pc_feat": item["pc_feat"],
            "pc_path": item["pc_path"],  # Strings are collated into lists
            "c_text": item["c_text"]
        }, s

def train_all():
    """
    Train over every file in the given base_paths
    """
    # ASSUMPTION: order is the same in both Folders, may need to change
    base_path_masked = "/home/bweiss/Benedikt/ShapeDream/data/dataset_masked/"
    
    # TODO: later change to whole folder
    train_samples = ["bag1.ply", "shoe1.ply", "shoe2.ply", "shoe3.ply", "shoe4.ply"]
    
    trainer = LoRATrainer(device='cuda', lora_rank=32, lora_alpha=8.0, num_steps=200)

    for sample in train_samples:
        # TODO: also need function to access shapeNet pointclouds
        #mesh = get_mesh_from_pc(sample)
        name = get_point_cloud_name_reg(sample)
        print("[TRAIN] NAME: ", name)  
        # define which GPU to use
        pointcloud_path_masked = base_path_masked #+ filename
        # Get pointNet++ features for masked input
        with torch.no_grad():
            pc_feat_dummy = get_pointnet_features(pointcloud_path=pointcloud_path_masked + sample, device=trainer.device)
        # train on sample
        trainer.train_with_point_cloud(pc_feat_dummy, pointcloud_path_masked + sample ,save_train_img=True, debug_name=name)
    return
    trainer.save_weights()


def train_all_interleaved():
    base_path_masked = "/home/bweiss/Benedikt/ShapeDream/data/dataset_masked/"
    train_samples = []
    
    num_samples = 256
    class_names = [#"airplane",
                   #"bag",
                   #"basket",
                   #"bathtub",
                   #"bed",
                   #"bench",
                   #"birdhouse",
                   #"bookshelf",
                   #"bottle",
                   #"bowl",
                   #"bus",
                   #"cabinet",
                   #"camera",
                   #"can",
                   #"cap",
                   #"car",
                   #"cellphone",
                   "chair",
                   #"clock",
                   #"dishwasher",
                   #"display",
                   #"earphone",
                   #"faucet",
                   #"file cabinet,
                   #"flowerpot",
                   #"guitar",
                   #"helmet",
                   #"jar",
                   #"keyboard",
                   #"knife",
                   #"lamp",
                   #"laptop",
                   #"loudspeaker",
                   #"mailbox",
                   #"microphone",
                   #"microwave",
                   #"motorbike",
                   #"mug",
                   #"piano",
                   #"pillow",
                   #"pistol",
                   #"printer",
                   #"remote",
                   #"rifle",
                   #"rocket",
                   #"skateboard",
                   #"sofa",
                   #"stove",
                   #"table",
                   #"telephone",
                   #"tower",
                   #"train",
                   #"trash bin",
                   #"washer",
                   #"watercraft",
                   ]

    for a in range(1, num_samples + 1):
        for cl in class_names:
            train_samples.append(f"shapenet_{cl}{a}.ply")
    #train_samples = ["shoe1.ply", "shoe2.ply", "shoe3.ply", "shoe4.ply"]

    val_samples = ["shapenet_chair210.ply", "shapenet_chair211.ply", "shapenet_chair212.ply", "shapenet_chair213.ply"
        ,"shapenet_chair214.ply", "shapenet_table220.ply"]
    
    #tester = Tester3D()

    trainer = LoRATrainer(
        device="cuda",
        lora_rank=32, # look if it works with 8
        lora_alpha=8.0,
        num_steps=800,     # not used by cached training, but keep for compatibility
        num_views=4,
        H=256,
        W=256,
        ELEV_DEG=15.0,
        DIST=2.5,
        load_from_ckpth=True,
        ckpt_path="checkpoints/mvdream_lora_pc_128_classes_chair_interleaved.pt"
    )


    train_cache = trainer.build_cache(
        train_samples=train_samples,
        base_path_masked=base_path_masked,
        save_debug_imgs=True,        # turn off if you don't want debug renders
        debug_dir="debug/cache/train",
        use_fixed_camera=True,
    )

    '''
    val_cache = trainer.build_cache(
        train_samples=val_samples,
        base_path_masked=base_path_masked,
        save_debug_imgs=True,
        debug_dir="debug/cache/val",
        use_fixed_camera=True,
    )
    '''

    # Save some memory by removing pointnet++
    trainer.pointnet = None
    num_epochs = 400

    val_every = 200                 # validate every N optimizer steps
    val_ddim_steps = 30             # keep small-ish for speed; use 50 if you can afford it
    val_scale = 7.5
    val_seed = 123
    val_num_samples = min(4, len(val_samples))  # evaluate on a subset each time

    best_val = float("inf")
    batch_size = 16

    train_dataset = CacheDataset(train_cache, train_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    global_step = 0
    pbar = tqdm(total=num_epochs * len(train_samples) // batch_size)

    for epoch in range(num_epochs):
        for batch_data, sample_name in train_loader:
            loss = trainer.train_one_step_from_cache(batch_data)

            sample = sample_name[0]
            """if step % val_every == 0 and step > 0:
                val_loss = trainer.compute_val_masked_img_loss(
                    val_cache=val_cache,
                    steps=val_ddim_steps,
                    scale=val_scale,
                    seed=val_seed,
                    num_samples=val_num_samples,
                )
                pbar.write(f"[VAL] step={step} masked_img_loss={val_loss:.6f}")
                if val_loss < best_val:
                    best_val = val_loss
                    trainer.save_weights("checkpoints/mvdream_lora_pc_best.pt")
                    pbar.write(f"[VAL] new best {best_val:.6f} -> saved")
            """
            pbar.set_description(f"step={global_step} sample={sample} train_loss={loss:.6f}")
            pbar.update(1)
            global_step += 1
    trainer.save_weights(f"checkpoints/mvdream_lora_pc_{num_samples}_classes_{class_names[0]}_interleaved_v2.pt")



if __name__ == "__main__":
    #overfit_bag()
    torch.set_float32_matmul_precision('medium')
    torch.manual_seed(42)
    train_all_interleaved()
    #train_all()