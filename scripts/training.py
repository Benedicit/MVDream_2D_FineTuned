import os
import random
import warnings

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
from dataloader import ShapeDreamDataModule


from trainer import LoRATrainer

warnings.filterwarnings("ignore", category=FutureWarning)

working_dir = os.path.dirname(os.path.abspath(__file__))

def train():
    L.seed_everything(42)  # replaces seed_everything(42) from __main__
    torch.set_float32_matmul_precision("medium")

    batch_size = 16
    num_samples = 16
    num_epochs = 850

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

    model = LoRATrainer(
        lora_rank=64,
        lora_alpha=8.0,
        num_views=4,
        H=256, W=256,
        ELEV_DEG=15.0,
        DIST=2.5,
        flow_matching=True,
        start_from_noise=True,
        load_from_ckpth=False,
        no_compile=False,
    )

    # Pass model_ref so DataModule can call build_cache with the right internals
    datamodule = ShapeDreamDataModule(
        class_names=class_names,
        num_samples=num_samples,
        batch_size=batch_size,
        debug_dir="debug/cache",
    )

    suffix = "flowmatching" if model.flow_matching else "diffusion"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"shapedream_{suffix}_utonia_{len(class_names)}_classes_{num_samples}_2l",
        monitor="val/mse",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Basically skip validation...
    val_every_n_steps = (num_samples * num_epochs) // batch_size - 1

    trainer = L.Trainer(
        max_epochs=num_epochs,
        precision="bf16-mixed",
        accelerator="gpu",
        devices=2,
        logger=TensorBoardLogger(save_dir="logs", name="two_self_attn_long"),
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=50)],
        log_every_n_steps=50,
        val_check_interval=val_every_n_steps,
    )

    # Pass datamodule instead of individual loaders —
    # Lightning will call datamodule.setup() before fit starts
    trainer.fit(model, datamodule=datamodule)



if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    train()