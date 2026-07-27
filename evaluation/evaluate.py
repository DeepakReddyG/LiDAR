"""evaluation/evaluate.py — score any labelled LAS against a hand-labelled GT
tile: per-class IoU + confusion matrix (MANUAL §6.0). Used to judge every
prompt/threshold/veto change, from the T3 rule baseline through the SAM3
pipeline.

Classes are keyed by LAS classification code, not by config.py class id:
pavement/sidewalk/parking share LAS code 11 (no standard LAS code tells them
apart), so they are necessarily one evaluation bucket regardless of how many
SAM3 prompts feed into it.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from config import CLASSES, UNLABELLED_LAS_CODE


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
    print("\nPer-class IoU:")
    for label, v in iou.items():
        print(f"  {label:<28s} {v:.3f}" if v == v else f"  {label:<28s}   n/a")

    print("\nConfusion matrix (rows=GT, cols=pred):")
    print(" " * 14 + "".join(f"{lb[:12]:>14s}" for lb in labels))
    for i, lb in enumerate(labels):
        print(f"{lb[:12]:>14s}" + "".join(f"{conf[i, j]:>14d}" for j in range(n)))

    return {"iou": iou, "confusion_matrix": conf.tolist(), "labels": labels}


def _make_synthetic_las(path: Path, codes: np.ndarray, orig_index: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="orig_index", type=np.uint32))
    las = laspy.LasData(header)
    n = len(codes)
    las.x = np.arange(n, dtype=np.float64)
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
