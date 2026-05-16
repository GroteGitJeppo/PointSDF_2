#!/usr/bin/env python3
"""Post-process organized CoRe++ RGB-D tree (intrinsics, split.json, dataset.json).

Usage::

    python data/setup_corepp_rgbd.py \\
        --dst data/3DPotatoTwin/corepp_rgbd \\
        --splits_csv data/3DPotatoTwin/splits.csv \\
        --intrinsics data/3DPotatoTwin/1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import pandas as pd


def _fanout_intrinsics(dst: str, intrinsics_src: str) -> int:
    n = 0
    for label in os.listdir(dst):
        label_dir = os.path.join(dst, label)
        if not os.path.isdir(label_dir):
            continue
        realsense_dir = os.path.join(label_dir, "realsense")
        if not os.path.isdir(realsense_dir):
            continue
        out_path = os.path.join(realsense_dir, "intrinsic.json")
        shutil.copy2(intrinsics_src, out_path)
        n += 1
    return n


def _write_split_json(dst: str, splits_csv: str) -> None:
    df = pd.read_csv(splits_csv)
    split_map: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split_name in split_map:
        labels = df.loc[df["split"] == split_name, "label"].astype(str).tolist()
        split_map[split_name] = sorted(set(labels))
    out_path = os.path.join(dst, "split.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2)
    print(f"Wrote {out_path} ({', '.join(f'{k}={len(v)}' for k, v in split_map.items())})")


def _write_dataset_json(dst: str, splits_csv: str) -> int:
    df = pd.read_csv(splits_csv)
    labels = sorted(df["label"].astype(str).unique())
    n = 0
    for label in labels:
        label_dir = os.path.join(dst, label)
        if not os.path.isdir(label_dir):
            continue
        path = os.path.join(label_dir, "dataset.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"is_useable": True}, f)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup CoRe++ RGB-D directory metadata")
    parser.add_argument("--dst", required=True, help="Organized corepp_rgbd root")
    parser.add_argument("--splits_csv", required=True, help="PointSDF splits.csv path")
    parser.add_argument("--intrinsics", required=True, help="Global RealSense intrinsic.json")
    parser.add_argument(
        "--skip_dataset_json",
        action="store_true",
        help="Do not write per-label dataset.json stubs",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dst):
        raise FileNotFoundError(args.dst)
    if not os.path.isfile(args.intrinsics):
        raise FileNotFoundError(args.intrinsics)

    n_k = _fanout_intrinsics(args.dst, args.intrinsics)
    print(f"Copied intrinsics to {n_k} labels")
    _write_split_json(args.dst, args.splits_csv)
    if not args.skip_dataset_json:
        n_d = _write_dataset_json(args.dst, args.splits_csv)
        print(f"Wrote dataset.json for {n_d} labels")


if __name__ == "__main__":
    main()
