import os
from PIL import Image as PilImage
import numpy as np


def save_training_views_grid(imgs, out_path, pad=16):
    """
    imgs: (V,3,H,W) in [-1,1]
    """
    imgs_np = (0.5 * (imgs + 1.0)).clamp(0,1)
    imgs_np = (imgs_np.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    V, H, W, C = imgs_np.shape
    canvas_h = H + 2 * pad
    canvas_w = V * W + (V + 1) * pad
    canvas = np.zeros((canvas_h, canvas_w, C), dtype=np.uint8)

    y = pad
    for i in range(V):
        x = pad + i * (W + pad)
        canvas[y:y + H, x:x + W, :] = imgs_np[i]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    PilImage.fromarray(canvas).save(out_path)
    print("Saved training views grid to", out_path)

