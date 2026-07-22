"""
main.py — LiDAR Point Cloud Classification Pipeline (v2, MANUAL.md)

v2 stages (design: MANUAL.md, build order: strategy.md):

    python main.py --stage features    # per-point ExG / intensity / returns
    python main.py --stage ground      # CSF ground filter → DTM → HAG
    python main.py --stage ortho       # RGB nadir ortho + stat grids + tiles
    python main.py --stage segment     # SAM3 text prompts → confidence grids
    python main.py --stage fuse        # thresholds + physics veto → label grid
    python main.py --stage map_back    # Z-aware labels → labelled.las
    python main.py --stage all         # all six, in order

    python main.py --stage baseline    # rule-only classifier on eval tiles
    python main.py --stage evaluate --gt <gt.las> --pred <pred.las>
"""

import argparse
import time
from pathlib import Path

from config import GRID_META_PATH, LAS_PATH


def _banner(text: str) -> None:
    print("\n" + "═" * 60 + f"\n{text}\n" + "═" * 60)


# ── v2 stage runners ─────────────────────────────────────────────────────────


def run_features() -> None:
    _banner("STAGE — features.py: per-point ExG / intensity / returns")
    from projection.features import run_features as run

    run()


def run_ground() -> None:
    _banner("STAGE — ground.py: CSF ground filter → DTM → per-point HAG")
    from projection.ground import run_ground as run

    run()


def run_ortho() -> None:
    _banner("STAGE — ortho.py: RGB nadir ortho + stat grids + tiles")
    from projection.ortho import run_ortho as run

    run()


def run_baseline() -> None:
    _banner("STAGE — baseline.py: rule-only classifier on eval tiles")
    from classification.baseline import run_baseline as run

    run()


def run_segment() -> None:
    _banner("STAGE — segment_sam3.py: SAM3 text prompts → confidence grids")
    from segmentation.segment_sam3 import run_segment_sam3

    run_segment_sam3()


def run_evaluate(gt_path: str, pred_path: str) -> None:
    _banner("STAGE — evaluate.py: per-class IoU + confusion matrix")
    from evaluation.evaluate import evaluate

    evaluate(gt_path, pred_path)


def run_fuse() -> None:
    _banner("STAGE — fuse.py: thresholds + physics veto + priority painting")
    from classification.fuse import run_fuse as run

    run()


def run_map_back() -> None:
    _banner("STAGE — map_back.py: Z-aware labels → 3D point cloud")
    from reprojection.map_back import run_map_back as run

    run()


# ── CLI ──────────────────────────────────────────────────────────────────────

V2_SEQUENCE = ["features", "ground", "ortho", "segment", "fuse", "map_back"]

STAGES = {
    "features": (run_features, [(LAS_PATH, "LAS input")]),
    "ground": (
        run_ground,
        [(LAS_PATH, "LAS input"), (GRID_META_PATH, "grid_meta.npz (run ortho first)")],
    ),
    "ortho": (
        run_ortho,
        [
            (LAS_PATH, "LAS input"),
            ("data/derived/exg.npy", "exg.npy (run features first)"),
            ("data/derived/hag.npy", "hag.npy (run ground first)"),
        ],
    ),
    "segment": (run_segment, [("data/slices/tiles", "tiles/ (run ortho first)")]),
    "fuse": (
        run_fuse,
        [
            ("data/masks/conf_pavement.npy", "conf grids (run segment first)"),
            ("data/slices/exg_grid.npy", "stat grids (run ortho first)"),
        ],
    ),
    "map_back": (
        run_map_back,
        [
            ("data/masks/label_grid.npy", "label_grid.npy (run fuse first)"),
            ("data/slices/surface_z.npy", "surface_z.npy (run ortho first)"),
        ],
    ),
    "baseline": (
        run_baseline,
        [("data/derived/hag.npy", "hag.npy (run ground first)")],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR classification pipeline (v2)")
    parser.add_argument("--stage", choices=[*STAGES, "evaluate", "all"], default="all")
    parser.add_argument("--gt", help="Ground-truth LAS path (--stage evaluate)")
    parser.add_argument("--pred", help="Predicted/labelled LAS path (--stage evaluate)")
    args = parser.parse_args()

    t_start = time.time()

    if args.stage == "evaluate":
        if not args.gt or not args.pred:
            raise SystemExit(
                "--stage evaluate requires --gt <gt.las> --pred <pred.las>"
            )
        _check_file(args.gt, "ground-truth LAS (crop + hand-label first)")
        _check_file(args.pred, "predicted/labelled LAS")
        run_evaluate(args.gt, args.pred)
        return

    names = V2_SEQUENCE if args.stage == "all" else [args.stage]
    for name in names:
        runner, prereqs = STAGES[name]
        for path, label in prereqs:
            _check_file(path, label)
        t0 = time.time()
        runner()
        print(f"  {name} done in {time.time() - t0:.1f}s")

    print(f"\nPipeline complete in {time.time() - t_start:.1f}s")


def _check_file(path, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}\n  → {label}")


if __name__ == "__main__":
    main()
