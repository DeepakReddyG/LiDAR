"""Check physical validity and label preservation at smoothing boundaries."""

import numpy as np
import pytest

from classification.fuse import _NAME_TO_ID, fuse


def scene():
    shape = (9, 9)
    return (
        {n: np.zeros(shape, np.float32) for n in _NAME_TO_ID},
        np.full(shape, 0.2, np.float32),
        np.full(shape, 0.2, np.float32),
        np.zeros(shape, bool),
    )


def test_grass_majority_cannot_replace_supported_tall_tree():
    conf, exg, hag, void = scene()
    conf["grass"][:] = 0.8
    hag[4, 4] = 10
    conf["tree"][4, 4] = 0.9
    first, _, _ = fuse(conf, exg, hag, void, tree_height_check=True)
    checked, confidence, _ = fuse(
        conf, exg, hag, void, tree_height_check=True, physical_smoothing=True
    )
    assert first[4, 4] == _NAME_TO_ID["grass"]
    assert checked[4, 4] == _NAME_TO_ID["tree"]
    assert confidence[4, 4] == conf["tree"][4, 4]


def test_unlabelled_majority_preserves_supported_grass():
    conf, exg, hag, void = scene()
    conf["grass"][4, 4] = 0.8
    default, _, _ = fuse(conf, exg, hag, void)
    checked, confidence, _ = fuse(conf, exg, hag, void, physical_smoothing=True)
    assert default[4, 4] == -1
    assert checked[4, 4] == _NAME_TO_ID["grass"]
    assert confidence[4, 4] == conf["grass"][4, 4]


@pytest.mark.parametrize("invalid", ["void", "unknown_height", "not_green"])
def test_neighbor_grass_cannot_fill_physically_invalid_unclaimed_pixel(invalid):
    conf, exg, hag, void = scene()
    conf["grass"][:] = 0.9
    if invalid == "void":
        void[4, 4] = True
    elif invalid == "unknown_height":
        hag[4, 4] = np.nan
    else:
        exg[4, 4] = -0.1
    checked, confidence, _ = fuse(conf, exg, hag, void, physical_smoothing=True)
    assert checked[4, 4] == -1
    assert confidence[4, 4] == 0


def test_physically_valid_neighbor_label_can_still_fill_hole():
    conf, exg, hag, void = scene()
    conf["grass"][:] = 0.9
    conf["grass"][4, 4] = 0.2  # Too weak to paint, but physically consistent.
    checked, confidence, _ = fuse(conf, exg, hag, void, physical_smoothing=True)
    assert checked[4, 4] == _NAME_TO_ID["grass"]
    assert confidence[4, 4] == conf["grass"][4, 4]


def test_combined_guard_preserves_low_grass_surrounded_by_tree():
    conf, exg, hag, void = scene()
    hag[:] = 10
    conf["tree"][:] = 0.9
    hag[4, 4] = 0.2
    conf["grass"][4, 4] = 0.8
    snapshots = [a.copy() for a in [*conf.values(), exg, hag, void]]
    checked, confidence, _ = fuse(
        conf, exg, hag, void, tree_height_check=True, physical_smoothing=True
    )
    assert checked[4, 4] == _NAME_TO_ID["grass"]
    assert confidence[4, 4] == conf["grass"][4, 4]
    assert checked[4, 3] == _NAME_TO_ID["tree"]
    for actual, original in zip([*conf.values(), exg, hag, void], snapshots):
        np.testing.assert_array_equal(actual, original)
