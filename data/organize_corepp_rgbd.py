#!/usr/bin/env python3
"""Convert 3DPotatoTwin ``1_rgbd/1_image`` layout to CoRe++ ``realsense/`` layout.

Usage::

    python data/organize_corepp_rgbd.py \\
        --src data/3DPotatoTwin/1_rgbd/1_image \\
        --dst data/3DPotatoTwin/corepp_rgbd

Then run :mod:`setup_corepp_rgbd` to add per-tuber intrinsics and split.json.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def organize(src: str, dst: str) -> int:
    n_frames = 0
    for potato_id in tqdm(sorted(os.listdir(src)), desc="Organize RGB-D"):
        id_folder = os.path.join(src, potato_id)
        if not os.path.isdir(id_folder):
            continue
        fnames = [f for f in os.listdir(id_folder) if "rgb" in f and f.endswith(".png")]
        for name in fnames:
            img_path = os.path.join(id_folder, name)
            dep_path = img_path.replace("_rgb_", "_depth_")
            if not os.path.isfile(dep_path):
                continue

            rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if rgba is None:
                continue
            img = rgba[:, :, :-1]
            mask = rgba[:, :, -1]

            dep = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
            if dep is None:
                continue

            dst_img_dir = os.path.join(dst, potato_id, "realsense", "color")
            dst_dep_dir = os.path.join(dst, potato_id, "realsense", "depth")
            dst_ann_dir = os.path.join(dst, potato_id, "realsense", "masks")
            _ensure_dir(dst_img_dir)
            _ensure_dir(dst_dep_dir)
            _ensure_dir(dst_ann_dir)

            dst_stem = os.path.splitext(name.replace("_rgb", ""))[0]
            with open(os.path.join(dst_dep_dir, dst_stem + ".npy"), "wb") as f:
                np.save(f, dep)
            cv2.imwrite(os.path.join(dst_ann_dir, dst_stem + ".png"), mask)
            cv2.imwrite(os.path.join(dst_img_dir, dst_stem + ".png"), img)
            n_frames += 1
    return n_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize 1_image RGB-D for CoRe++ loader")
    parser.add_argument(
        "--src",
        default="data/3DPotatoTwin/1_rgbd/1_image",
        help="Source root with <label>/*_rgb_*.png folders",
    )
    parser.add_argument(
        "--dst",
        default="data/3DPotatoTwin/corepp_rgbd",
        help="Output root with <label>/realsense/{color,depth,masks}/",
    )
    args = parser.parse_args()
    n = organize(args.src, args.dst)
    print(f"Wrote {n} frames under {args.dst}")


if __name__ == "__main__":
    main()
