"""
main.py — LiDAR Point Cloud Segmentation Pipeline

Full pipeline:
    .las → slice.py → segment.py → classify.py → map_back.py

Usage
-----
Run full pipeline (from project root, with .venv active):

    python main.py

Or run individual stages:

    python main.py --stage slice
    python main.py --stage segment
    python main.py --stage classify
    python main.py --stage map_back

Skip regenerating the slice if top_down.png already exists:

    python main.py --skip-slice

Use a trained classifier instead of the heuristic:

    python main.py --stage classify   # after you've run classify.train()
"""

import argparse
import time
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

LAS_PATH        = "data/raw/UPark_Merged_PS_NAD83_G18_USFT_las.las"
SLICE_PATH      = "data/slices/top_down.png"
GRID_PATH       = "data/slices/elevation_grid.npy"
GRID_META_PATH  = "data/slices/grid_meta.npz"
MASKS_PATH      = "data/masks/masks.npy"
META_PATH       = "data/masks/mask_meta.npy"
LABEL_GRID_PATH = "data/masks/label_grid.npy"
WEIGHTS_PATH    = "models/classifier/vgg19_head.pt"
DEVICE          = "mps"        # Apple Silicon; change to "cuda" or "cpu"
RESOLUTION      = 0.5          # ft per pixel


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_slice() -> None:
    print("\n" + "═" * 60)
    print("STAGE 1 — slice.py: Rasterise LAS → top-down PNG")
    print("═" * 60)
    t0 = time.time()
    from src.slice import load_point_cloud, project_top_down
    xy, z = load_point_cloud(LAS_PATH)
    grid, meta = project_top_down(xy, z, resolution=RESOLUTION, output_path=SLICE_PATH)
    print(f"  slice done in {time.time() - t0:.1f}s")


def run_segment() -> None:
    print("\n" + "═" * 60)
    print("STAGE 2 — segment.py: SAM2 automatic mask generation")
    print("═" * 60)
    t0 = time.time()
    from src.segment import run_segmentation
    masks, meta = run_segmentation(
        image_path=SLICE_PATH,
        out_dir="data/masks",
        device=DEVICE,
    )
    print(f"  segment done in {time.time() - t0:.1f}s  →  {len(masks)} masks")


def run_classify() -> None:
    print("\n" + "═" * 60)
    print("STAGE 3 — classify.py: Per-mask classification")
    print("═" * 60)
    t0 = time.time()
    from src.classify import predict
    class_labels, label_grid = predict(
        image_path=SLICE_PATH,
        grid_path=GRID_PATH,
        masks_path=MASKS_PATH,
        meta_path=META_PATH,
        weights_path=WEIGHTS_PATH,
        out_dir="data/masks",
        device=DEVICE,
    )
    print(f"  classify done in {time.time() - t0:.1f}s")


def run_map_back() -> None:
    print("\n" + "═" * 60)
    print("STAGE 4 — map_back.py: Labels → 3D point cloud")
    print("═" * 60)
    t0 = time.time()
    from src.map_back import map_labels_to_points
    labels = map_labels_to_points(
        las_path=LAS_PATH,
        grid_meta_path=GRID_META_PATH,
        label_grid_path=LABEL_GRID_PATH,
        out_dir="data/output",
    )
    print(f"  map_back done in {time.time() - t0:.1f}s  →  {len(labels):,} labelled points")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiDAR segmentation pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["slice", "segment", "classify", "map_back", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--skip-slice",
        action="store_true",
        help="Skip slice.py if top_down.png already exists",
    )
    args = parser.parse_args()

    stage = args.stage
    t_start = time.time()

    if stage in ("slice", "all"):
        if args.skip_slice and Path(SLICE_PATH).exists():
            print(f"\n[skip] {SLICE_PATH} already exists — skipping slice stage")
            print("       (remove --skip-slice to regenerate)")
        else:
            _check_file(LAS_PATH, "LAS input")
            run_slice()

    if stage in ("segment", "all"):
        _check_file(SLICE_PATH, "top_down.png (run slice stage first)")
        run_segment()

    if stage in ("classify", "all"):
        _check_file(MASKS_PATH, "masks.npy (run segment stage first)")
        _check_file(GRID_PATH,  "elevation_grid.npy (run slice stage first)")
        run_classify()

    if stage in ("map_back", "all"):
        _check_file(LABEL_GRID_PATH, "label_grid.npy (run classify stage first)")
        _check_file(GRID_META_PATH,  "grid_meta.npz (run slice stage first)")
        run_map_back()

    print(f"\n{'═' * 60}")
    print(f"Pipeline complete in {time.time() - t_start:.1f}s")
    print(f"{'═' * 60}")
    print("Outputs:")
    print(f"  data/slices/top_down.png          false-colour 2D slice")
    print(f"  data/masks/masks.npy              SAM2 segment masks")
    print(f"  data/masks/labelled_overlay.png   class visualisation")
    print(f"  data/output/labelled_points.npz   X, Y, Z, label per point")
    print(f"  data/output/labelled.las           labelled LAS file")


def _check_file(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"  → {label}"
        )


if __name__ == "__main__":
    main()
