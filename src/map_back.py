"""
map_back.py — Project 2D pixel class labels back to the original 3D point cloud.

Pipeline position:  classify.py → [map_back.py]

Uses the same grid coordinate mapping as slice.py so every 3D point is
assigned the class label of the 2D pixel it projects into.

Inputs:
    data/raw/<file>.las               original LAS file
    data/slices/grid_meta.npz         coordinate metadata (from slice.py)
    data/masks/label_grid.npy         (H, W) class-per-pixel (from classify.py)

Outputs:
    data/output/labelled_points.npz   X, Y, Z, label arrays (one entry per point)
    data/output/labelled.las          copy of original .las with classification field

Class indices (must match classify.py):
    0  road_surface
    1  bridge_deck
    2  guardrail
    3  vegetation
    4  vehicle
   -1  background / unlabelled
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────────────

LAS_PATH        = "data/raw/UPark_Merged_PS_NAD83_G18_USFT_las.las"
GRID_META_PATH  = "data/slices/grid_meta.npz"
LABEL_GRID_PATH = "data/masks/label_grid.npy"
OUT_DIR         = "data/output"

CLASS_NAMES = ["road_surface", "bridge_deck", "guardrail", "vegetation", "vehicle"]

# LAS classification codes to write (standard LAS 1.4 codes + custom range)
# Standard: 0=unclassified, 1=unassigned, 2=ground, 11-255 custom
LABEL_TO_LAS_CLASS = {
   -1:  0,   # unclassified
    0:  2,   # road_surface → LAS "ground"
    1: 17,   # bridge_deck  → LAS "bridge deck"
    2: 14,   # guardrail    → LAS "wire guard" (closest standard)
    3:  4,   # vegetation   → LAS "medium vegetation"
    4:  6,   # vehicle      → LAS "building" (fallback; no standard vehicle code)
}


def map_labels_to_points(
    las_path:        str = LAS_PATH,
    grid_meta_path:  str = GRID_META_PATH,
    label_grid_path: str = LABEL_GRID_PATH,
    out_dir:         str = OUT_DIR,
    chunk_size:      int = 10_000_000,
) -> np.ndarray:
    """
    Assign a class label to every 3D point by projecting it into the 2D label grid.

    Processes the LAS file in chunks to keep memory usage bounded
    (411 M points × 4 bytes ≈ 1.6 GB for labels alone).

    Parameters
    ----------
    chunk_size : int
        Number of points to process at once (default 10M ≈ ~120 MB per chunk).

    Returns
    -------
    labels : (N,) int32  — class label per point (−1 = background / outside grid)
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # ── Load grid metadata ────────────────────────────────────────────────────
    meta = np.load(grid_meta_path)
    x_min      = float(meta["x_min"])
    y_max      = float(meta["y_max"])
    resolution = float(meta["resolution"])
    rows       = int(meta["rows"])
    cols       = int(meta["cols"])
    print(f"Grid: {rows}×{cols} at {resolution} ft/px")
    print(f"  X_min={x_min:.2f}  Y_max={y_max:.2f}")

    # ── Load label grid ───────────────────────────────────────────────────────
    label_grid = np.load(label_grid_path)   # (H, W) int32
    print(f"Label grid shape: {label_grid.shape}")

    # ── Open LAS ─────────────────────────────────────────────────────────────
    las = laspy.read(las_path)
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    N = len(x)
    print(f"Points: {N:,}")

    # ── Project points → grid indices → labels ────────────────────────────────
    print("Projecting points to grid …")
    col_idx = np.clip(
        ((x - x_min) / resolution).astype(np.int32), 0, cols - 1
    )
    row_idx = np.clip(
        ((y_max - y) / resolution).astype(np.int32), 0, rows - 1
    )

    labels = label_grid[row_idx, col_idx].astype(np.int32)

    # ── Print class distribution ──────────────────────────────────────────────
    print("\nPoint-cloud class distribution:")
    for cls_id, name in enumerate(CLASS_NAMES):
        count = int((labels == cls_id).sum())
        pct   = 100 * count / N
        print(f"  {cls_id}  {name:<14s}  {count:>12,} pts  ({pct:.2f}%)")
    unlab = int((labels == -1).sum())
    print(f" -1  background     {unlab:>12,} pts  ({100*unlab/N:.2f}%)")

    # ── Save compressed numpy output ─────────────────────────────────────────
    npz_path = out_dir_p / "labelled_points.npz"
    np.savez_compressed(
        npz_path,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        z=np.array(las.z, dtype=np.float32),
        label=labels,
    )
    print(f"\nSaved labelled_points.npz → {npz_path}")

    # ── Write labelled LAS ────────────────────────────────────────────────────
    _write_labelled_las(las, labels, out_dir_p / "labelled.las")

    return labels


def _write_labelled_las(
    source_las: laspy.LasData,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    """
    Write a new LAS file with the classification field set to our class codes.
    Preserves all original point attributes.
    """
    # Build LAS class code array  (uint8 field in LAS spec)
    las_cls = np.vectorize(LABEL_TO_LAS_CLASS.get)(labels).astype(np.uint8)

    # Create output LAS with same point format
    out_las = laspy.LasData(header=source_las.header)
    out_las.points = source_las.points.copy()

    # Overwrite the classification field
    out_las.classification = las_cls

    out_las.write(str(out_path))
    print(f"Saved labelled.las → {out_path}")
    print(f"  Point format: {source_las.header.point_format.id}")
    print(f"  LAS class codes used: {sorted(set(las_cls.tolist()))}")


def load_labelled_points(
    npz_path: str = f"{OUT_DIR}/labelled_points.npz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience loader for downstream analysis.

    Returns
    -------
    x, y, z : (N,) float32
    labels  : (N,) int32
    """
    data = np.load(npz_path)
    return data["x"], data["y"], data["z"], data["label"]


if __name__ == "__main__":
    labels = map_labels_to_points(
        las_path=LAS_PATH,
        grid_meta_path=GRID_META_PATH,
        label_grid_path=LABEL_GRID_PATH,
        out_dir=OUT_DIR,
    )
    print(f"\nDone!  Labelled {len(labels):,} points.")
