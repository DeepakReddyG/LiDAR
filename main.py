"""
main.py — LiDAR Point Cloud Classification Pipeline (v2, MANUAL.md)

v2 stages (design: MANUAL.md, build order: strategy.md):

    python main.py --stage features    # per-point ExG / intensity / returns
    python main.py --stage ground      # CSF ground filter → DTM → HAG
    python main.py --stage ortho       # RGB nadir ortho + stat grids + tiles
    python main.py --stage baseline    # rule-only classifier on eval tiles
    python main.py --stage evaluate --gt <gt.las> --pred <pred.las>
    python main.py --stage all         # features → ground → ortho → baseline

v1 stages, kept until T5/T6 replace them:

    python main.py --stage segment     # SAM2 automatic masks (superseded by T5)
    python main.py --stage classify    # heuristic classifier  (superseded by T6)
    python main.py --stage map_back    # column map-back        (rewritten at T6)
"""

import argparse
import time
from pathlib import Path

from config import GRID_META_PATH, LAS_PATH

# v1 paths (die with the remaining v1 stages)
SLICE_PATH      = "data/slices/top_down.png"
GRID_PATH       = "data/slices/elevation_grid.npy"
MASKS_PATH      = "data/masks/masks.npy"
META_PATH       = "data/masks/mask_meta.npy"
LABEL_GRID_PATH = "data/masks/label_grid.npy"
WEIGHTS_PATH    = "models/classifier/vgg19_head.pt"
DEVICE          = "mps"


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


def run_evaluate(gt_path: str, pred_path: str) -> None:
    _banner("STAGE — evaluate.py: per-class IoU + confusion matrix")
    from evaluation.evaluate import evaluate
    evaluate(gt_path, pred_path)


# ── v1 stage runners (superseded at T5/T6) ───────────────────────────────────

def run_segment() -> None:
    _banner("STAGE [v1] — segment.py: SAM2 automatic mask generation")
    from segmentation.segment import run_segmentation
    run_segmentation(image_path=SLICE_PATH, out_dir="data/masks", device=DEVICE)


def run_classify() -> None:
    _banner("STAGE [v1] — classify.py: per-mask classification")
    from classification.classify import predict
    predict(image_path=SLICE_PATH, grid_path=GRID_PATH, masks_path=MASKS_PATH,
            meta_path=META_PATH, weights_path=WEIGHTS_PATH,
            out_dir="data/masks", device=DEVICE)


def run_map_back() -> None:
    _banner("STAGE [v1] — map_back.py: labels → 3D point cloud")
    from reprojection.map_back import map_labels_to_points
    map_labels_to_points(las_path=str(LAS_PATH), grid_meta_path=str(GRID_META_PATH),
                         label_grid_path=LABEL_GRID_PATH, out_dir="data/output")


# ── CLI ──────────────────────────────────────────────────────────────────────

V2_SEQUENCE = ["features", "ground", "ortho", "baseline"]

STAGES = {
    "features": (run_features, [(LAS_PATH, "LAS input")]),
    "ground":   (run_ground,   [(LAS_PATH, "LAS input"),
                                (GRID_META_PATH, "grid_meta.npz (run ortho or v1 slice first)")]),
    "ortho":    (run_ortho,    [(LAS_PATH, "LAS input"),
                                ("data/derived/exg.npy", "exg.npy (run features first)"),
                                ("data/derived/hag.npy", "hag.npy (run ground first)")]),
    "baseline": (run_baseline, [("data/derived/hag.npy", "hag.npy (run ground first)")]),
    "segment":  (run_segment,  [(SLICE_PATH, "top_down.png (v1 slice output)")]),
    "classify": (run_classify, [(MASKS_PATH, "masks.npy (run segment first)"),
                                (GRID_PATH, "elevation_grid.npy (v1 slice output)")]),
    "map_back": (run_map_back, [(LABEL_GRID_PATH, "label_grid.npy (run classify first)"),
                                (GRID_META_PATH, "grid_meta.npz")]),
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
            raise SystemExit("--stage evaluate requires --gt <gt.las> --pred <pred.las>")
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
