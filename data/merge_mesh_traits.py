#!/usr/bin/env python3
"""
Merge mesh_traits_2023.csv and mesh_traits_2025.csv into mesh_traits.csv.

Does not modify the source files. Adds column `year`. 2025 rows get empty
group_id, bucket_id, pin_id.

Usage (from PointSDF_2/):
    python data/merge_mesh_traits.py
"""

from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent / "3DPotatoTwin"
PATH_2023 = DIR / "mesh_traits_2023.csv"
PATH_2025 = DIR / "mesh_traits_2025.csv"
OUT = DIR / "mesh_traits.csv"

TRAITCOLS = [
    "minor axis length (cm)",
    "middle axis length (cm)",
    "major axis length (cm)",
    "area (cm2)",
    "volume (cm3)",
    "aspect ratio",
    "volume/surface ratio",
    "sphericity",
    "convexity",
]


def main() -> None:
    df23 = pd.read_csv(PATH_2023)
    df25 = pd.read_csv(PATH_2025)

    if set(df23["label"]) & set(df25["label"]):
        raise SystemExit(
            f"Duplicate labels between sources: {set(df23['label']) & set(df25['label'])}"
        )

    missing_t = [c for c in TRAITCOLS if c not in df23.columns or c not in df25.columns]
    if missing_t:
        raise SystemExit(f"Missing trait columns: {missing_t}")

    # 2023: label, year, group_id, bucket_id, pin_id, traits...
    df23o = df23.copy()
    df23o.insert(1, "year", 2023)

    # 2025: same column order; missing IDs left empty
    df25o = pd.DataFrame(
        {
            "label": df25["label"],
            "year": 2025,
            "group_id": "",
            "bucket_id": "",
            "pin_id": "",
        }
    )
    for c in TRAITCOLS:
        df25o[c] = df25[c].values

    # Ensure column parity
    if list(df23o.columns) != list(df25o.columns):
        raise SystemExit(
            f"Column mismatch:\n  2023: {list(df23o.columns)}\n  2025: {list(df25o.columns)}"
        )

    out = pd.concat([df23o, df25o], ignore_index=True)

    if out["label"].duplicated().any():
        raise SystemExit("Duplicate label in merged frame.")

    if len(out) != len(df23) + len(df25):
        raise SystemExit("Row count mismatch after merge.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows): {len(df23)} from 2023, {len(df25)} from 2025.")


if __name__ == "__main__":
    main()
