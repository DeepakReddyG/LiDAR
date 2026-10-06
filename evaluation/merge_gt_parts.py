"""evaluation/merge_gt_parts.py — assemble a GT tile from per-class LAS parts
cut in CloudCompare (MANUAL §6.0 step 3).

CloudCompare edits classification values awkwardly but segments and saves
sub-clouds well, so the hand-labelling workflow is: cut each class into its
own file, then let this script stamp the codes.

    data/eval/gt_parts/tile_c_tree.las      -> LAS code 5
    data/eval/gt_parts/tile_c_grass.las     -> LAS code 3
    ...                                     -> data/eval/tile_c_gt.las

Points not present in any part stay UNLABELLED_LAS_CODE. The evaluator ignores
GT codes 0 and 1 after matching points, so partial labelling is valid. Errors
on annotated points still count; scores describe only the annotated portion.

First restore CloudCompare exports with evaluation.annotation. Restored parts
are matched to the full tile by orig_index, never by file order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import laspy
import numpy as np

from config import ANNOTATION_XYZ_TOLERANCE, CLASSES, EVAL_DIR, UNLABELLED_LAS_CODE

# Part filename suffix -> LAS code. pavement/sidewalk/parking all map to 11,
# so either name works for a hard-surface cut.
_SUFFIX_TO_CODE = {info["name"]: info["las_code"] for info in CLASSES.values()}


def merge_gt_parts(
    tile: str, parts_dir: Path | None = None, eval_dir: Path | None = None
) -> Path:
    """Stamp per-class part files onto `tile_{tile}.las` -> `tile_{tile}_gt.las`."""
    eval_dir = eval_dir or EVAL_DIR
    parts_dir = parts_dir or eval_dir / "gt_parts"
    tile_path = eval_dir / f"tile_{tile}.las"
    if not tile_path.exists():
        raise SystemExit(f"{tile_path} missing — run evaluation/crop_tiles.py first")
    out = eval_dir / f"tile_{tile}_gt.las"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing ground truth: {out}")

    parts = sorted(parts_dir.glob(f"tile_{tile}_*.las"))
    if not parts:
        raise SystemExit(
            f"No parts found at {parts_dir}/tile_{tile}_<class>.las\n"
            f"Expected suffixes: {', '.join(sorted(_SUFFIX_TO_CODE))}"
        )

    las = laspy.read(tile_path)
    if "orig_index" not in las.point_format.dimension_names:
        raise SystemExit(f"{tile_path} has no orig_index — recrop with crop_tiles.py")
    tile_idx = np.asarray(las.orig_index)
    if not len(tile_idx):
        raise ValueError("Reference tile contains no points")
    if len(np.unique(tile_idx)) != len(tile_idx):
        raise ValueError("Reference tile has duplicate orig_index values")
    order = np.argsort(tile_idx)

    codes = np.full(len(tile_idx), UNLABELLED_LAS_CODE, np.uint8)
    for part in parts:
        suffix = part.stem[len(f"tile_{tile}_") :]
        code = _SUFFIX_TO_CODE.get(suffix)
        if code is None:
            print(f"  [skip] {part.name}: '{suffix}' is not a known class — ignored")
            continue

        p = laspy.read(part)
        if "orig_index" not in p.point_format.dimension_names:
            raise SystemExit(
                f"{part.name} has no orig_index — run evaluation.annotation restore "
                "on the CloudCompare export before merging"
            )
        p_idx = np.asarray(p.orig_index)
        if len(np.unique(p_idx)) != len(p_idx):
            raise ValueError(f"{part.name}: duplicate orig_index values")

        # Locate each part point in the tile by orig_index (sorted searchsorted).
        pos = np.searchsorted(tile_idx, p_idx, sorter=order)
        pos = np.clip(pos, 0, len(order) - 1)
        hit = order[pos]
        found = tile_idx[hit] == p_idx
        if not found.all():
            raise ValueError(
                f"{part.name}: {(~found).sum():,} IDs not in reference tile"
            )

        for dim in ("x", "y", "z"):
            if not np.allclose(
                np.asarray(getattr(las, dim))[hit],
                np.asarray(getattr(p, dim)),
                rtol=0,
                atol=ANNOTATION_XYZ_TOLERANCE,
            ):
                raise ValueError(f"{part.name}: coordinates disagree with orig_index")

        target = hit[found]
        clash = (codes[target] != UNLABELLED_LAS_CODE) & (codes[target] != code)
        if clash.any():
            raise ValueError(
                f"{part.name}: {clash.sum():,} points have conflicting classes"
            )
        codes[target] = code
        print(f"  {part.name:<34s} -> code {code:<3d} {found.sum():>10,} points")

    las.classification = codes
    if not np.any(codes != UNLABELLED_LAS_CODE):
        raise ValueError("No annotated points found in recognized class parts")
    with out.open("xb") as stream:
        las.write(stream)

    labelled = (codes != UNLABELLED_LAS_CODE).sum()
    print(
        f"\n{out}: {labelled:,}/{len(codes):,} points labelled ({labelled / len(codes):.1%})"
    )
    return out


def _self_check() -> None:
    import tempfile

    from evaluation.evaluate import _make_synthetic_las

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = tmp / "gt_parts"
        parts.mkdir()

        # Tile of 100 points; parts cover 0-29 (grass) and 60-79 (tree),
        # deliberately shuffled and partial.
        n = 100
        idx = np.arange(n)
        _make_synthetic_las(tmp / "tile_z.las", np.full(n, 1, np.uint8), idx)
        perm = np.random.RandomState(0).permutation(30)
        _make_synthetic_las(
            parts / "tile_z_grass.las", np.zeros(30, np.uint8), idx[:30][perm]
        )
        _make_synthetic_las(
            parts / "tile_z_tree.las", np.zeros(20, np.uint8), idx[60:80]
        )

        merged = laspy.read(merge_gt_parts("z", parts, eval_dir=tmp))
        got_by_idx = np.asarray(merged.classification)[
            np.argsort(np.asarray(merged.orig_index))
        ]

        assert (got_by_idx[:30] == 3).all(), "grass part not stamped"
        assert (got_by_idx[30:60] == UNLABELLED_LAS_CODE).all(), (
            "gap not left unlabelled"
        )
        assert (got_by_idx[60:80] == 5).all(), "tree part not stamped"
        assert (got_by_idx[80:] == UNLABELLED_LAS_CODE).all(), (
            "tail not left unlabelled"
        )

    print("self-check OK: parts stamped by orig_index, unlabelled gaps preserved")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
        raise SystemExit

    # With no args, merge every tile that actually has parts — partial
    # labelling is the normal state, so a missing tile is not an error.
    tiles = sys.argv[1:] or sorted(
        {p.stem.split("_")[1] for p in (EVAL_DIR / "gt_parts").glob("tile_*_*.las")}
    )
    if not tiles:
        raise SystemExit(
            f"No parts under {EVAL_DIR / 'gt_parts'}/tile_<tile>_<class>.las — "
            f"cut them in CloudCompare first"
        )
    for tile in tiles:
        print(f"\nTile {tile}:")
        merge_gt_parts(tile)
