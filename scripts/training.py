import os
import warnings

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from dataloader import ShapeDreamDataModule
from trainer import LoRATrainer
import torch.distributed as dist

warnings.filterwarnings("ignore", category=FutureWarning)

working_dir = os.path.dirname(os.path.abspath(__file__))

import torch._dynamo
torch._dynamo.config.optimize_ddp = False

def train():
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    num_samples = 2000
    batch_size = 16
    num_epochs = 1000

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
        model=model.model,
        num_samples=num_samples,
        batch_size=batch_size,
        debug_dir="debug/cache",
    )

    suffix = "flowmatching" if model.flow_matching else "diffusion"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"shapedream_{suffix}_utonia_{len(class_names)}_classes_{num_samples}_2l_dist",
        monitor=None,
        save_weights_only=True,
        every_n_epochs=min(num_epochs, 50),
        save_top_k=1,
        save_last=False,
    )

    val_every_n_steps = (num_samples * 20) // batch_size

    trainer = L.Trainer(
        max_epochs=num_epochs,
        precision="bf16-mixed",
        accelerator="gpu",
        devices="auto",
        logger=TensorBoardLogger(save_dir="logs", name="two_self_attn_long"),
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=5)],
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        # Skip validation...
        limit_val_batches=0.0,
        reload_dataloaders_every_n_epochs=0,
        strategy=DDPStrategy(
            static_graph=True,
            find_unused_parameters=False),
        #val_check_interval=val_every_n_steps,
    )

    try:
        trainer.fit(model, datamodule=datamodule)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    train()