"""evaluation/evaluate.py — score any labelled LAS against a hand-labelled GT
tile: per-class IoU + confusion matrix (MANUAL §6.0). Used to judge every
prompt/threshold/veto change, from the T3 rule baseline through the SAM3
pipeline.

Classes are keyed by LAS classification code, not by config.py class id:
pavement/sidewalk/parking share LAS code 11 (no standard LAS code tells them
apart), so they are necessarily one evaluation bucket regardless of how many
SAM3 prompts feed into it.

After matching points, GT codes 0 and 1 are excluded from all scoring. An
unlabelled prediction on annotated GT is still an error. No score is produced
when none of the matched GT points are annotated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import laspy
import numpy as np

from config import CLASSES, EVAL_IGNORE_GT_CODES, UNLABELLED_LAS_CODE


def _code_names(classes: dict = CLASSES) -> dict[int, str]:
    """LAS code -> display name, merging classes that share a code."""
    names: dict[int, list[str]] = {}
    for info in classes.values():
        names.setdefault(info["las_code"], []).append(info["name"])
    return {code: "/".join(ns) for code, ns in names.items()}


def _match_by_orig_index(gt: laspy.LasData, pred: laspy.LasData):
    gt_idx = np.asarray(gt.orig_index)
    pred_idx = np.asarray(pred.orig_index)
    common, gi, pi = np.intersect1d(gt_idx, pred_idx, return_indices=True)
    return gi, pi, len(common)


def _match_by_xyz(gt: laspy.LasData, pred: laspy.LasData):
    def key(las):
        xyz = np.round(np.stack([las.x, las.y, las.z], axis=1), 2)
        return xyz.view([("x", "f8"), ("y", "f8"), ("z", "f8")]).reshape(-1)

    common, gi, pi = np.intersect1d(key(gt), key(pred), return_indices=True)
    return gi, pi, len(common)


def evaluate(gt_path: str | Path, pred_path: str | Path) -> dict:
    gt = laspy.read(gt_path)
    pred = laspy.read(pred_path)

    has_index = (
        "orig_index" in gt.point_format.dimension_names
        and "orig_index" in pred.point_format.dimension_names
    )
    if has_index:
        gi, pi, n_common = _match_by_orig_index(gt, pred)
    else:
        print("WARNING: orig_index missing — falling back to rounded-XYZ matching")
        gi, pi, n_common = _match_by_xyz(gt, pred)
    if n_common == 0:
        raise ValueError("No matching points between GT and prediction")

    gt_codes = np.asarray(gt.classification)[gi]
    pred_codes = np.asarray(pred.classification)[pi]
    annotated = ~np.isin(gt_codes, EVAL_IGNORE_GT_CODES)
    n_scored = int(annotated.sum())
    n_ignored = n_common - n_scored
    if n_scored == 0:
        raise ValueError(
            "No annotated ground-truth points among matched points "
            f"(GT codes {EVAL_IGNORE_GT_CODES} are ignored)"
        )
    gt_codes = gt_codes[annotated]
    pred_codes = pred_codes[annotated]

    names = _code_names()
    codes = sorted(
        set(names)
        | {UNLABELLED_LAS_CODE}
        | set(gt_codes.tolist())
        | set(pred_codes.tolist())
    )
    labels = [
        names.get(c, "unlabelled" if c == UNLABELLED_LAS_CODE else f"code_{c}")
        for c in codes
    ]
    code_to_row = {c: i for i, c in enumerate(codes)}

    n = len(codes)
    conf = np.zeros((n, n), dtype=np.int64)
    gi2 = np.vectorize(code_to_row.get)(gt_codes)
    pi2 = np.vectorize(code_to_row.get)(pred_codes)
    np.add.at(conf, (gi2, pi2), 1)

    iou = {}
    for c, label in zip(codes, labels):
        i = code_to_row[c]
        tp = int(conf[i, i])
        fp = int(conf[:, i].sum()) - tp
        fn = int(conf[i, :].sum()) - tp
        union = tp + fp + fn
        iou[label] = float(tp / union) if union > 0 else float("nan")

    print(
        f"Matched {n_common:,} points ({'orig_index' if has_index else 'XYZ fallback'})"
    )
    print(f"Scored {n_scored:,} annotated points; ignored {n_ignored:,} GT points")
    print("\nPer-class IoU:")
    for label, v in iou.items():
        print(
            f"  {label:<28s} {v:.3f}" if not math.isnan(v) else f"  {label:<28s}   n/a"
        )

    print("\nConfusion matrix (rows=GT, cols=pred):")
    print(" " * 14 + "".join(f"{lb[:12]:>14s}" for lb in labels))
    for i, lb in enumerate(labels):
        print(f"{lb[:12]:>14s}" + "".join(f"{conf[i, j]:>14d}" for j in range(n)))

    return {
        "iou": iou,
        "confusion_matrix": conf.tolist(),
        "labels": labels,
        "matched_points": n_common,
        "scored_points": n_scored,
        "ignored_gt_points": n_ignored,
    }


def _pred_from_labels_npy(tile_path: Path, out_path: Path, labels_npy: Path) -> Path:
    """Stamp pipeline class ids from labels.npy onto an eval tile via orig_index."""
    from reprojection.map_back import labels_to_las_codes

    tile = laspy.read(tile_path)
    idx = np.asarray(tile.orig_index)
    all_labels = np.load(labels_npy, mmap_mode="r")
    tile.classification = labels_to_las_codes(np.asarray(all_labels[idx]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile.write(str(out_path))
    return out_path


def run_evaluate_all() -> dict:
    """Score every eval tile that has a GT against baseline and (if present) labels.npy."""
    from config import EVAL_DIR, EVAL_TILE_BOUNDS, OUTPUT_DIR

    baseline_scores: dict = {}
    sam3_scores: dict = {}
    n_gt = 0
    labels_npy = OUTPUT_DIR / "labels.npy"

    for name in EVAL_TILE_BOUNDS:
        gt_path = EVAL_DIR / f"tile_{name}_gt.las"
        if not gt_path.exists():
            print(
                f"  [no GT] {gt_path.name} missing — hand-label in CloudCompare "
                f"(see evaluation/crop_tiles.py instructions / REMAINING_FIXES_PLAN Task H1)"
            )
            continue
        n_gt += 1

        baseline_path = EVAL_DIR / f"tile_{name}_baseline.las"
        if baseline_path.exists():
            print(f"\nBaseline tile {name}:")
            baseline_scores[name] = evaluate(gt_path, baseline_path)["iou"]
        else:
            print(
                f"  [skip baseline] {baseline_path.name} missing — run --stage baseline"
            )

        tile_path = EVAL_DIR / f"tile_{name}.las"
        if labels_npy.exists() and tile_path.exists():
            pred_path = EVAL_DIR / f"tile_{name}_sam3.las"
            print(f"\nSAM3 tile {name} (from labels.npy via orig_index):")
            _pred_from_labels_npy(tile_path, pred_path, labels_npy)
            sam3_scores[name] = evaluate(gt_path, pred_path)["iou"]
        elif not labels_npy.exists():
            print(f"  [skip SAM3] {labels_npy} missing — run --stage map_back")

    if n_gt == 0:
        raise SystemExit(
            "No ground-truth tiles found under data/eval/tile_*_gt.las.\n"
            "Hand-label the cropped tiles in CloudCompare first (Task H1)."
        )

    if baseline_scores:
        out = EVAL_DIR / "baseline_scores.json"
        out.write_text(json.dumps(baseline_scores, indent=2))
        print(f"\nSaved {out}")
    if sam3_scores:
        out = EVAL_DIR / "sam3_scores.json"
        out.write_text(json.dumps(sam3_scores, indent=2))
        print(f"Saved {out}")

    return {"baseline": baseline_scores, "sam3": sam3_scores}


def _make_synthetic_las(path: Path, codes: np.ndarray, orig_index: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="orig_index", type=np.uint32))
    las = laspy.LasData(header)
    n = len(codes)
    # Identity-based coordinates keep shuffled/subset fixtures physically aligned.
    las.x = orig_index.astype(np.float64)
    las.y = np.zeros(n, dtype=np.float64)
    las.z = np.zeros(n, dtype=np.float64)
    las.classification = codes.astype(np.uint8)
    las.orig_index = orig_index.astype(np.uint32)
    las.write(str(path))


def _self_check() -> None:
    import tempfile

    # 100 synthetic points: 50 pavement (code 11), 50 grass (code 3).
    n = 100
    gt_codes = np.array([11] * 50 + [3] * 50)
    gt_index = np.arange(n)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gt_path = tmp / "gt.las"
        _make_synthetic_las(gt_path, gt_codes, gt_index)

        # Perfect prediction, shuffled file order — orig_index must still align.
        perm = np.random.RandomState(0).permutation(n)
        perfect_path = tmp / "perfect.las"
        _make_synthetic_las(perfect_path, gt_codes[perm], gt_index[perm])

        result = evaluate(gt_path, perfect_path)
        assert abs(result["iou"]["pavement/sidewalk/parking"] - 1.0) < 1e-9
        assert abs(result["iou"]["grass"] - 1.0) < 1e-9

        # Known corruption: flip 10 grass points (orig_index 50-59) to pavement.
        corrupt_codes = gt_codes.copy()
        corrupt_codes[50:60] = 11
        corrupt_path = tmp / "corrupt.las"
        _make_synthetic_las(corrupt_path, corrupt_codes, gt_index)

        result = evaluate(gt_path, corrupt_path)
        # pavement: TP=50, FP=10, FN=0 -> 50/60 ; grass: TP=40, FP=0, FN=10 -> 40/50
        assert abs(result["iou"]["pavement/sidewalk/parking"] - 50 / 60) < 1e-9
        assert abs(result["iou"]["grass"] - 40 / 50) < 1e-9

    print(
        "self-check OK: perfect-match IoU=1.0, known-corruption IoU matches hand calc"
    )


if __name__ == "__main__":
    _self_check()
