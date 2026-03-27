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
from lora import add_lora_to_cross_att_only, LoRALinear
from pointnet_encoder import read_from_plyfile, get_pointnet_features, PointFeatProjector, get_point_cloud_name_reg
from view_renderer import PointRenderer
from tqdm import tqdm
from trainer import LoRATrainer, get_mesh_from_pc
from tester import Tester3D
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

working_dir = os.path.dirname(os.path.abspath(__file__))

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
            "c_text": item["c_text"],
            "pc_latent": item["pc_latent"],
        }, s


def train_all_interleaved():
    base_path_masked = f"{working_dir}/../../data/dataset_masked/"
    train_samples = []
    
    num_samples = 16
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
        if a >= 600:
            a += 20
        for cl in class_names:
            train_samples.append(f"shapenet_{cl}{a}.ply")
    #train_samples = ["shoe1.ply", "shoe2.ply", "shoe3.ply", "shoe4.ply"]

    val_samples = ["shapenet_chair250.ply", "shapenet_chair251.ply", "shapenet_chair252.ply", "shapenet_chair253.ply"
        ,"shapenet_chair254.ply", "shapenet_table250.ply", "shapenet_lamp250.ply", "shapenet_bench250.ply", "shapenet_car250.ply"]
    
    #tester = Tester3D()

    trainer = LoRATrainer(
        device="cuda",
        lora_rank=64, # look if it works with 8
        lora_alpha=8.0,
        num_views=4,
        H=256,
        W=256,
        ELEV_DEG=15.0,
        DIST=2.5,
        flow_matching=True,
        load_from_ckpth=False,
        #ckpt_path="checkpoints/mvdream_lora_pc_128_classes_chair_interleaved.pt"
    )

    train_cache = trainer.build_cache(
        train_samples=train_samples,
        base_path_masked=base_path_masked,
        save_target_imgs=False,        # turn off if you don't want debug renders
        save_pc_imgs=False,        # turn off if you don't want debug renders
        debug_dir="debug/cache/train",
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
    num_epochs = 1000

    val_every = 200                 # validate every N optimizer steps
    val_ddim_steps = 30             # keep small-ish for speed; use 50 if you can afford it
    val_scale = 7.5
    val_seed = 123
    val_num_samples = min(4, len(val_samples))  # evaluate on a subset each time

    best_val = float("inf")
    batch_size = 16

    train_dataset = CacheDataset(train_cache, train_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
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
    suffix = "flowmatching" if trainer.flow_matching else "diffusion"
    trainer.save_weights(f"checkpoints/shapedream_{suffix}_{len(class_names)}_classes_{num_samples}_obj.pt")



if __name__ == "__main__":
    #overfit_bag()
    torch.set_float32_matmul_precision('medium')
    torch.manual_seed(42)
    train_all_interleaved()
    #train_all()