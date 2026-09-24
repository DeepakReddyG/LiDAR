"""classification/baseline.py — rule-only baseline classifier (MANUAL §8 step 4, T3).

Five numpy comparisons on per-point features (exg, hag), no neural net.
This is the bar the SAM3 pipeline must beat; the gap between them is a
reportable result either way.

Rules differ deliberately from the MANUAL §6.5 veto table: those thresholds
veto SAM3 *claims* (appearance already says "tree"); standalone they misfire —
exg alone turns every lawn into a tree, hag alone every grey tower. Standalone
tree needs height AND greenness. First claim wins, most-specific first.

Classifies the eval tiles only (via their orig_index stamp into the
data/derived/ memmaps) — a full-site baseline.las would cost a 16 GB write
and add nothing measurable. Scores against tile_*_gt.las when present.
"""

from __future__ import annotations

import json
import math

import laspy
import numpy as np

from config import (
    BASELINE,
    CLASSES,
    DERIVED_DIR,
    EVAL_DIR,
    EVAL_TILE_BOUNDS,
    UNLABELLED_LAS_CODE,
)

_NAME_TO_ID = {info["name"]: cid for cid, info in CLASSES.items()}


def classify(exg: np.ndarray, hag: np.ndarray) -> np.ndarray:
    """Rule-only labels: config class ids, -1 = unlabelled. First claim wins."""
    b = BASELINE
    labels = np.full(exg.shape, -1, dtype=np.int32)

    rules = [
        ("building", (hag > b["building_hag"]) & (exg <= b["hard_exg"])),
        ("tree", (hag > b["tree_hag"]) & (exg > b["veg_exg"])),
        ("grass", (hag < b["grass_hag"]) & (exg > b["veg_exg"])),
        ("pavement", (hag < b["pavement_hag"]) & (exg <= b["hard_exg"])),
    ]
    for name, mask in rules:
        labels[(labels == -1) & mask] = _NAME_TO_ID[name]
    return labels


def label_tile(tile_path, out_path) -> np.ndarray:
    """Classify one cropped eval tile via its orig_index stamp; write a copy
    with the baseline classification codes."""
    tile = laspy.read(tile_path)
    idx = np.asarray(tile.orig_index)
    exg = np.load(DERIVED_DIR / "exg.npy", mmap_mode="r")[idx]
    hag = np.load(DERIVED_DIR / "hag.npy", mmap_mode="r")[idx]

    labels = classify(exg, hag)
    id_to_code = {cid: info["las_code"] for cid, info in CLASSES.items()}
    id_to_code[-1] = UNLABELLED_LAS_CODE
    tile.classification = np.vectorize(id_to_code.get)(labels).astype(np.uint8)
    tile.write(str(out_path))

    n = len(labels)
    print(
        f"  {out_path.name}: {n:,} pts — "
        + "  ".join(
            f"{name}:{(labels == cid).sum() / n:.1%}"
            for name, cid in _NAME_TO_ID.items()
            if (labels == cid).any()
        )
        + f"  unlabelled:{(labels == -1).sum() / n:.1%}"
    )
    return labels


def run_baseline() -> None:
    scores = {}
    for name in EVAL_TILE_BOUNDS:
        tile_path = EVAL_DIR / f"tile_{name}.las"
        out_path = EVAL_DIR / f"tile_{name}_baseline.las"
        if not tile_path.exists():
            raise FileNotFoundError(f"{tile_path} — run evaluation/crop_tiles.py first")
        label_tile(tile_path, out_path)

        gt_path = EVAL_DIR / f"tile_{name}_gt.las"
        if gt_path.exists():
            from evaluation.evaluate import evaluate

            print(f"\nScoring tile {name} against {gt_path.name}:")
            scores[name] = evaluate(gt_path, out_path)["iou"]
        else:
            print(
                f"  [no GT yet] {gt_path.name} missing — label it in CloudCompare to score"
            )

    if scores:
        out = EVAL_DIR / "baseline_scores.json"
        out.write_text(json.dumps(scores, indent=2))
        print(f"\nSaved {out}")
        for name, iou in scores.items():
            assert any(v > 0 for v in iou.values() if not math.isnan(v)), (
                f"tile {name}: all IoU zero — rules or matching wired wrong"
            )


def _self_check() -> None:
    """Each rule branch + priority on crafted feature pairs."""
    exg = np.array([0.02, 0.20, 0.20, 0.02, 0.20, 0.02, 0.30])
    hag = np.array([10.0, 10.0, 0.5, 0.5, 4.0, 4.0, 10.0])
    #               bldg  tree  grass pave  none  none  tree(green+tall)
    got = classify(exg, hag)
    want = [
        _NAME_TO_ID["building"],
        _NAME_TO_ID["tree"],
        _NAME_TO_ID["grass"],
        _NAME_TO_ID["pavement"],
        -1,
        -1,
        _NAME_TO_ID["tree"],
    ]
    assert got.tolist() == want, f"{got.tolist()} != {want}"
    print("self-check OK: all rule branches + priority")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self-check", action="store_true")
    if p.parse_args().self_check:
        _self_check()
    else:
        run_baseline()
