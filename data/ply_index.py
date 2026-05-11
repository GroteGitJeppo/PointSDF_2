"""
PLY file index — fast lookup instead of rglob on network filesystems.

Build once on the server:
    python -m data.ply_index --data_root data/3DPotatoTwin/1_rgbd/2_pcd \
                             --output    data/3DPotatoTwin/ply_index.csv

Then set  ply_index_csv: data/3DPotatoTwin/ply_index.csv  in your encoder config.
test.py and select_checkpoint.py will use the CSV for O(1) lookup instead of
crawling the filesystem.  Falls back to rglob if the key is absent or the file
does not exist, so the change is fully backward-compatible.
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def load_ply_files(
    data_root: str,
    split_ids: set[str],
    index_csv: str | None = None,
) -> list[str]:
    """Return PLY paths for the given label IDs.

    Fast path: read a pre-built CSV with columns ``label``, ``ply_path``.
    Slow path: rglob the filesystem (used when index_csv is None / missing).
    """
    if index_csv and os.path.exists(index_csv):
        df = pd.read_csv(index_csv, dtype={'label': str})
        paths = df.loc[df['label'].isin(split_ids), 'ply_path'].tolist()
        return paths

    # Fallback — warns so the user knows to build the index
    print(
        'WARNING: ply_index_csv not set or file not found — '
        'falling back to rglob (slow on network filesystems). '
        'Run  python -m data.ply_index  to build the index.'
    )
    all_files = list(Path(data_root).rglob('*.ply'))
    return [str(f) for f in all_files if f.parent.name in split_ids]


def build_ply_index(data_root: str, output_csv: str) -> int:
    """Walk data_root once and write a label → ply_path index CSV.

    Returns the number of PLY files indexed.
    """
    rows = []
    for f in Path(data_root).rglob('*.ply'):
        rows.append({'label': f.parent.name, 'ply_path': str(f)})

    df = pd.DataFrame(rows, columns=['label', 'ply_path'])
    df.sort_values(['label', 'ply_path'], inplace=True, ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return len(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build a PLY file index CSV for fast lookup in test.py / select_checkpoint.py.'
    )
    parser.add_argument(
        '--data_root', required=True,
        help='Root directory to scan (e.g. data/3DPotatoTwin/1_rgbd/2_pcd)',
    )
    parser.add_argument(
        '--output', required=True,
        help='Path to write the CSV (e.g. data/3DPotatoTwin/ply_index.csv)',
    )
    args = parser.parse_args()

    n = build_ply_index(args.data_root, args.output)
    print(f'Indexed {n} PLY files → {args.output}')
