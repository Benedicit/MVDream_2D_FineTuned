import os
import warnings
from datetime import timedelta

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from dataloader import ShapeDreamDataModule
from scripts.dataloader import PCNDataModule
from trainer import LoRATrainer
import torch.distributed as dist

warnings.filterwarnings("ignore", category=FutureWarning)

working_dir = os.path.dirname(os.path.abspath(__file__))

import torch._dynamo
torch._dynamo.config.optimize_ddp = False

def train(ckpt_path=None):
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    num_samples = 4000
    batch_size = 32
    num_epochs = 120
    #steps_per_epoch = 530
    steps_per_epoch = 605
    #steps_per_epoch = 504
    #steps_per_epoch = 333


    print(f"Batch size: {batch_size}, Number Epochs: {num_epochs}")
    class_names = ["airplane",
                   "bag",
                   "basket",
                   "bathtub",
                   "bed",
                   "bench",
                   "birdhouse",
                   "bookshelf",
                   "bottle",
                   "bowl",
                   "bus",
                   "cabinet",
                   "camera",
                   "can",
                   "cap",
                   "car",
                   "cellphone",
                   "chair",
                   "clock",
                   "dishwasher",
                   "display",
                   "earphone",
                   "faucet",
                   "file cabinet",
                   "flowerpot",
                   "guitar",
                   "helmet",
                   "jar",
                   "keyboard",
                   "knife",
                   "lamp",
                   "laptop",
                   "loudspeaker",
                   "mailbox",
                   "microphone",
                   "microwave",
                   "motorbike",
                   "mug",
                   "piano",
                   "pillow",
                   "pistol",
                   "printer",
                   "remote",
                   "rifle",
                   "rocket",
                   "skateboard",
                   "sofa",
                   "stove",
                   "table",
                   "telephone",
                   "tower",
                   "train",
                   "trash bin",
                   "washer",
                   "watercraft",
                   ]

    model = LoRATrainer(
        #lora_rank=80,
        lora_rank=32,
        #lora_alpha=20.0,
        lora_alpha=8.0,
        num_views=4,
        H=256, W=256,
        ELEV_DEG=15.0,
        DIST=2.5,
        flow_matching=True,
        start_from_noise=True,
        no_compile=False,
        lr=5e-5,
        batch_size=batch_size,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        use_text=False,
    )

    # Pass model_ref so DataModule can call build_cache with the right internals
    """
    datamodule = ShapeDreamDataModule(
        class_names=class_names,
        model=model.model,
        num_samples=num_samples,
        batch_size=batch_size,
        debug_dir="debug/cache",
    )
    """
    datamodule = PCNDataModule(
        model=model.model,
        batch_size=batch_size,
        debug_dir="debug/cache",
    )

    suffix = "flowmatching" if model.flow_matching else "diffusion"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        #filename=f"shapedream_{suffix}_{len(class_names)}_classes_{num_samples}_big_proj",
        filename=f"shapedream_32_lora",
        monitor=None,
        save_weights_only=False,
        every_n_epochs=min(num_epochs, 10),
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(
        max_epochs=num_epochs,
        precision="bf16-mixed",
        accelerator="gpu",
        devices="auto",
        logger=TensorBoardLogger(save_dir="logs", name="paper"),
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=20), lr_monitor],
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        # Skip validation...
        limit_val_batches=0.0,
        check_val_every_n_epoch=10,
        reload_dataloaders_every_n_epochs=0,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        strategy=DDPStrategy(
            static_graph=True,
            find_unused_parameters=False,
            timeout=timedelta(days=3.0)),
        #val_check_interval=val_every_n_steps,
    )

    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    train(ckpt_path=None)
    #train(ckpt_path="checkpoints/shapedream_pcn-v4.ckpt")

