from PIL import Image
import numpy as np

im = Image.open("rgb_000.png")
arr = np.array(im)  # should be (H,W,4)

a = arr[..., 3]
rgb = arr[..., :3]

print("mode:", im.mode, "shape:", arr.shape)
print("alpha>0 ratio:", (a > 0).mean())
print("mean RGB where alpha==0:", rgb[a == 0].mean(axis=0) if (a==0).any() else None)
