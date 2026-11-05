"""Small synthetic Hydra MP3D-format dataset for testing the fault injectors."""
import argparse
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

# (label, row0, row1, col0, col1) in fractions of the image; two chairs so blobs repeat per class
BOXES = [(3, .45, .80, .05, .20), (3, .50, .85, .75, .90), (5, .55, .90, .30, .55),
         (7, .20, .55, .60, .72), (22, .15, .35, .10, .25), (14, .60, .95, .58, .66)]
# camera optical frame (z forward, y down) expressed in a z-up body frame
R_BODY_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def make_dataset(root, n=30, h=120, w=160, seed=0, depth_dtype="float32", label_dtype="uint8"):
    rng = np.random.default_rng(seed)
    root = Path(root)
    for d in ("color", "depth", "labels"):
        (root / d).mkdir(parents=True, exist_ok=True)
    f = 0.8 * w
    with open(root / "camera_info.yaml", "w") as fo:
        yaml.safe_dump({"cx": w / 2.0, "cy": h / 2.0, "fx": f, "fy": f, "height": h, "width": w}, fo)

    t = np.arange(n)
    yaw = 0.4 * np.sin(t / 6.0)
    pos = np.stack([0.1 * t, 0.3 * np.sin(t / 8.0), np.full(n, 1.2)], axis=1)
    quat = Rotation.from_matrix(Rotation.from_euler("z", yaw).as_matrix() @ R_BODY_CAM).as_quat()
    with open(root / "poses.csv", "w", newline="") as fo:
        fo.write("timestamp_ns,tx,ty,tz,qw,qx,qy,qz\n")
        for i in range(n):
            x, y, z = map(float, pos[i])
            qx, qy, qz, qw = map(float, quat[i])
            fo.write(f"{i * 200_000_000},{x!r},{y!r},{z!r},{qw!r},{qx!r},{qy!r},{qz!r}\n")

    rows = np.arange(h)[:, None] / h
    for i in range(n):
        lab = np.where(rows < .25, 17, np.where(rows < .75, 1, 2)) * np.ones((h, w), int)
        depth = 3.0 + 2.0 * np.arange(w)[None, :] / w + 0.01 * i + np.zeros((h, 1))
        for k, (c, r0, r1, c0, c1) in enumerate(BOXES):
            shift = int(i * 0.5) * (1 if k % 2 else -1)
            ys, xs = slice(int(r0 * h), int(r1 * h)), slice(max(0, int(c0 * w) + shift), max(1, int(c1 * w) + shift))
            lab[ys, xs] = c
            depth[ys, xs] = 1.5 + 0.1 * k
        lab[rng.random((h, w)) < 0.01] = 0
        depth = depth + rng.normal(0, 0.01, (h, w))
        depth[rng.random((h, w)) < 0.01] = np.nan
        rgb = (lab[..., None] * np.array([7, 13, 29]) + rng.integers(0, 30, (h, w, 3))).astype(np.uint8)
        Image.fromarray(rgb).save(root / "color" / f"rgb_{i:07d}.png")
        Image.fromarray(lab.astype(label_dtype)).save(root / "labels" / f"labels_{i:07d}.png")
        if depth_dtype == "uint16":
            depth = np.where(np.isnan(depth), 0, np.rint(depth * 1000)).astype(np.uint16)
        tifffile.imwrite(root / "depth" / f"depth_{i:07d}.tiff", depth.astype(depth_dtype))
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--depth-dtype", default="float32", choices=["float32", "uint16"])
    a = ap.parse_args()
    make_dataset(a.root, a.frames, depth_dtype=a.depth_dtype)
