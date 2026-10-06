"""Wires each pipeline module's existing `--self-check` block into pytest so
CI actually runs them (main.py --stage X --self-check equivalents).

features.py's self-check reads the source LAS file (~16GB, not present on
CI runners) — it is skipped there and runs locally when the file exists."""

import pytest

from config import LAS_PATH


def test_grid_self_check():
    from projection.grid import _self_check

    _self_check()


def test_ground_self_check():
    from projection.ground import _self_check

    _self_check()


def test_ortho_self_check():
    from projection.ortho import _self_check

    _self_check()


@pytest.mark.skipif(
    not LAS_PATH.exists(), reason="requires the source LAS file, not present in CI"
)
def test_features_self_check():
    from projection.features import _self_check

    _self_check()


def test_baseline_self_check():
    from classification.baseline import _self_check

    _self_check()


def test_fuse_self_check():
    from classification.fuse import _self_check

    _self_check()


def test_map_back_self_check():
    from reprojection.map_back import _self_check

    _self_check()


def test_segment_sam3_self_check():
    from segmentation.segment_sam3 import _self_check

    _self_check()


def test_evaluate_self_check():
    from evaluation.evaluate import _self_check

    _self_check()


def test_fuse_priority_order_isolated():
    """FUSE_PRIORITY: most-specific-first, first claim wins. A whole region
    (not a single pixel, so the majority filter can't erase it as speckle)
    claimed identically by two classes must resolve to whichever comes first
    in config.FUSE_PRIORITY — here "sidewalk" before "pavement"."""
    import numpy as np

    from classification.fuse import _NAME_TO_ID, fuse
    from config import FUSE_PRIORITY

    assert FUSE_PRIORITY.index("sidewalk") < FUSE_PRIORITY.index("pavement"), (
        "test assumes sidewalk outranks pavement in FUSE_PRIORITY"
    )

    shape = (20, 20)
    z = np.zeros(shape, dtype=np.float32)
    exg = np.full(shape, 0.0, np.float32)  # hard surface everywhere
    hag = np.full(shape, 0.5, np.float32)  # low HAG, passes the pavement veto
    void = np.zeros(shape, bool)

    # Every pixel claimed by both "sidewalk" and "pavement" at equal confidence.
    conf = {n: z.copy() for n in _NAME_TO_ID}
    conf["pavement"][:, :] = 0.9
    conf["sidewalk"][:, :] = 0.9

    label, _, _ = fuse(conf, exg, hag, void)
    assert np.all(label == _NAME_TO_ID["sidewalk"]), (
        "lower-priority class won a contested region"
    )


def test_merge_gt_parts_self_check():
    from evaluation.merge_gt_parts import _self_check

    _self_check()
