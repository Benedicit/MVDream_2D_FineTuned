import math
import os
import random
import torch
import torch.nn.functional as F
from lightning import seed_everything
from torch.utils.tensorboard import SummaryWriter

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
            "pc_path": item["pc_path"],
            "c_text": item["c_text"],
            "pc_feat": item["pc_feat"],
            "camera": item["camera"],
            "training_points": item["training_points"],
            "training_normals": item["training_normals"],
        }, s


def train_all_interleaved():
    log_dir = "logs/check_overfitting"
    writer = SummaryWriter(log_dir=log_dir)

    base_path_masked = f"{working_dir}/../../data/dataset_masked/"
    train_samples = []
    
    num_samples = 1280
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

    val_samples = []
    for a in range(num_samples + 50, num_samples + max(10, int(num_samples * 0.1)) + 1):
        if a >= 600:
            a += 20
        for cl in class_names:
            val_samples.append(f"shapenet_{cl}{a}.ply")
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
        start_from_noise=False,
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
        save_target_imgs=False,
        save_pc_imgs=False,
        debug_dir="debug/cache/val",
    )
    '''

    # Save some memory by removing pointnet++
    trainer.pointnet = None
    num_epochs = 500

    val_every = 200                 # validate every N optimizer steps
    val_steps = 30                  # ODE steps during validation inference
    val_scale = 7.5
    val_seed = 123
    val_num_samples = min(8, len(val_samples))  # evaluate on a subset each time

    best_val_mse = float("inf")
    batch_size = 16

    train_dataset = CacheDataset(train_cache, train_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    global_step = 0
    pbar = tqdm(total=num_epochs * len(train_samples) // batch_size)

    suffix = "flowmatching" if trainer.flow_matching else "diffusion"

    for epoch in range(num_epochs):
        for batch_data, sample_name in train_loader:
            loss = trainer.train_one_step_from_cache(batch_data)

            writer.add_scalar("Loss/train", loss, global_step)

            sample = sample_name[0]

            '''
            if global_step % val_every == 0:
                val_metrics = trainer.validation_step(
                    val_cache=val_cache,
                    num_samples=val_num_samples,
                    steps=val_steps,
                    cfg_scale=val_scale,
                    seed=val_seed,
                )
                pbar.write(
                    f"[VAL] step={global_step}  "
                    f"mse={val_metrics['mse']:.6f}  "
                    #f"psnr={val_metrics['psnr']:.2f} dB"
                )
                writer.add_scalar("Metrics/MSE", val_metrics["mse"], global_step)
                if val_metrics["mse"] < best_val_mse:
                    best_val_mse = val_metrics["mse"]
                    pbar.write(f"[VAL] New best MSE: {best_val_mse:.2f} — checkpoint saved.")
            '''

            pbar.set_description(f"step={global_step} sample={sample} train_loss={loss:.6f}")
            pbar.update(1)
            global_step += 1

    trainer.save_weights(f"checkpoints/shapedream_{suffix}_latent_{len(class_names)}_classes_{num_samples}.pt")
    writer.close()


if __name__ == "__main__":
    #overfit_bag()
    torch.set_float32_matmul_precision('medium')
    seed_everything(42)
    train_all_interleaved()
    #train_all()