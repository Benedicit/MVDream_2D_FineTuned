import os
import random
import warnings

import torch
from lightning import seed_everything
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from trainer import LoRATrainer

warnings.filterwarnings("ignore", category=FutureWarning)
#torch._dynamo.config.capture_scalar_outputs = True

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
            "pc_feat": item["pc_feat_x_split"] if random.random() > 0.5 else item["pc_feat_z_split"],
        }, s

def collate_pc_feat(batch):
    items, sample_ids = zip(*batch)
    pc_feats = [item["pc_feat"] for item in items]
    lengths = torch.tensor([x.shape[0] for x in pc_feats], dtype=torch.long)
    flat_pc_feat = torch.cat(pc_feats, dim=0)
    batch_idx = torch.repeat_interleave(
        torch.arange(len(pc_feats), device=flat_pc_feat.device),
        lengths,
    )
    pc_feat_dense, pc_feat_mask = to_dense_batch(flat_pc_feat, batch_idx)

    z = torch.stack([item["z"] for item in items], dim=0)
    pc_path = [item["pc_path"] for item in items]
    c_text = torch.stack([item["c_text"] for item in items], dim=0)
    return {
        "z": z,
        "pc_path": pc_path,
        "c_text": c_text,
        "pc_feat": pc_feat_dense,
        "pc_feat_mask": pc_feat_mask,
    }, list(sample_ids)


def train_all_interleaved():
    log_dir = "logs/two_self_attn_long"
    writer = SummaryWriter(log_dir=log_dir)

    base_path_masked = f"{working_dir}/../../data/dataset_masked/"
    train_samples = []
    
    num_samples = 2000
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

    batch_size = 16

    val_samples = []
    num_val_samples = int(num_samples * 0.1)
    num_val_samples += num_val_samples % batch_size
    for a in range(num_samples + 50, num_samples + 50 + max(batch_size, num_val_samples)):
        if a >= 600:
            a += 20
        for cl in class_names:
            val_samples.append(f"shapenet_{cl}{a}.ply")
    #tester = Tester3D()

    trainer = LoRATrainer(
        device="cuda",
        lora_rank=64,
        lora_alpha=8.0,
        num_views=4,
        H=256,
        W=256,
        ELEV_DEG=15.0,
        DIST=2.5,
        flow_matching=True,
        start_from_noise=True,
        load_from_ckpth=False,
        #ckpt_path="checkpoints/mvdream_lora_pc_128_classes_chair_interleaved.pt"
        no_compile=False,
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

    num_epochs = 850

    val_every = len(train_samples) * 20  # validate every N epochs
    val_steps = 25                       # ODE steps during validation inference
    val_scale = 7.5

    best_val_mse = float("inf")

    train_dataset = CacheDataset(train_cache, train_samples)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_pc_feat)
    '''
    validation_data = CacheDataset(val_cache, val_samples)
    val_loader = DataLoader(validation_data,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=collate_pc_feat)
    '''
    global_step = 0
    pbar = tqdm(total=num_epochs * len(train_samples) // batch_size)

    suffix = "flowmatching" if trainer.flow_matching else "diffusion"

    for epoch in range(num_epochs):
        for batch_data, sample_name in train_loader:
            loss = trainer.train_one_step_from_cache(batch_data)

            writer.add_scalar("Loss/train", loss, global_step)

            sample = sample_name[0]

            if global_step % val_every == 0 and 1 == 0:
                mse = 0.0
                l1 = 0.0
                for batch_val, _ in val_loader:
                    val_metrics = trainer.validation_step(
                        batch=batch_val,
                        steps=val_steps,
                        cfg_scale=val_scale,
                    )
                    batch_mse = val_metrics["mse"]
                    batch_l1 = val_metrics["l1"]
                    pbar.write(
                        f"[VAL] step={global_step}  "
                        f"MSE(latent)={batch_mse:.6f}  "
                        f"L1(img)={batch_l1:.6f}"
                    )
                    writer.add_scalar("Metrics/MSE_Batch", batch_mse, global_step)
                    writer.add_scalar("Metrics/L1_Batch", batch_l1, global_step)
                    mse += val_metrics["mse"]
                    l1 += val_metrics["l1"]
                mse = mse / len(val_loader)
                l1 = l1 / len(val_loader)
                writer.add_scalar("Metrics/MSE_Avg", mse, global_step)
                writer.add_scalar("Metrics/L1_Avg", l1, global_step)
                if mse < best_val_mse:
                    best_val_mse = mse
                    pbar.write(f"[VAL] New best MSE: {best_val_mse:.2f}")

            pbar.set_description(f"step={global_step} sample={sample} train_loss={loss:.6f}")
            pbar.update(1)
            global_step += 1

    trainer.save_weights(f"checkpoints/shapedream_{suffix}_utonia_{len(class_names)}_classes_{num_samples}_2l.pt")
    writer.close()


if __name__ == "__main__":
    #overfit_bag()
    torch.set_float32_matmul_precision('medium')
    seed_everything(42)
    train_all_interleaved()
    #train_all()