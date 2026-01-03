import numpy as np
from pathlib import Path

def inspect_cams(cam_dir):
    cam_dir = Path(cam_dir)
    print(cam_dir)
    for p in sorted(cam_dir.glob("cam_*.txt")):
        lines = p.read_text().splitlines()
        ext = np.array([[float(x) for x in lines[i].split()] for i in range(1,5)], dtype=np.float32)  # w2c
        R = ext[:3,:3]
        t = ext[:3,3]
        C = -R.T @ t  # camera center in world
        az = np.degrees(np.arctan2(C[2], C[0]))
        print(p.name, "C=", C, "az≈", az)

inspect_cams("/home/temp_compute/Benedikt/ShapeDream/mvdream_2D/debug/bag/")