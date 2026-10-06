"""Regressions for the opt-in tree height and smoothing consistency experiment."""

import numpy as np
import pytest

from classification.fuse import _NAME_TO_ID, fuse
from config import VETO


def scene():
    shape = (9, 9)
    conf = {name: np.zeros(shape, np.float32) for name in _NAME_TO_ID}
    return (
        conf,
        np.full(shape, 0.2, np.float32),
        np.full(shape, 10.0, np.float32),
        np.zeros(shape, bool),
    )


def test_low_green_tree_claim_falls_back_to_supported_grass():
    conf, exg, hag, void = scene()
    hag[:] = 0.2
    conf["tree"][:] = 0.9
    conf["grass"][:] = 0.8
    legacy, _, _ = fuse(conf, exg, hag, void)
    checked, confidence, stats = fuse(conf, exg, hag, void, tree_height_check=True)
    assert np.all(legacy == _NAME_TO_ID["tree"])
    assert np.all(checked == _NAME_TO_ID["grass"])
    np.testing.assert_array_equal(confidence, conf["grass"])
    assert stats["tree"]["kept_px"] == 0


def test_smoothing_cannot_overwrite_supported_low_grass_with_tree():
    conf, exg, hag, void = scene()
    conf["tree"][:] = 0.9
    conf["grass"][4, 4] = 0.8
    hag[4, 4] = 0.2
    checked, confidence, _ = fuse(conf, exg, hag, void, tree_height_check=True)
    assert checked[4, 4] == _NAME_TO_ID["grass"]
    assert confidence[4, 4] == conf["grass"][4, 4]
    assert checked[4, 3] == _NAME_TO_ID["tree"]


@pytest.mark.parametrize("height", [0.2, VETO["tree_hag"], np.nan, np.inf])
def test_rejected_tree_without_alternative_stays_unlabelled(height):
    conf, exg, hag, void = scene()
    conf["tree"][:] = 0.9
    hag[4, 4] = height
    checked, confidence, _ = fuse(conf, exg, hag, void, tree_height_check=True)
    assert checked[4, 4] == -1
    assert confidence[4, 4] == 0


def test_smoothing_cannot_reintroduce_tree_on_void():
    conf, exg, hag, void = scene()
    conf["tree"][:] = 0.9
    void[4, 4] = True
    checked, confidence, _ = fuse(conf, exg, hag, void, tree_height_check=True)
    assert checked[4, 4] == -1
    assert confidence[4, 4] == 0


def test_tall_tree_and_inputs_are_preserved():
    conf, exg, hag, void = scene()
    conf["tree"][:] = 0.9
    # Tall, non-green canopy still passes the existing height evidence.
    exg[:] = -0.1
    snapshots = [a.copy() for a in [*conf.values(), exg, hag, void]]
    legacy, old_conf, _ = fuse(conf, exg, hag, void)
    checked, confidence, _ = fuse(conf, exg, hag, void, tree_height_check=True)
    np.testing.assert_array_equal(checked, legacy)
    np.testing.assert_array_equal(confidence, old_conf)
    assert np.all(checked == _NAME_TO_ID["tree"])
    for actual, original in zip([*conf.values(), exg, hag, void], snapshots):
        np.testing.assert_array_equal(actual, original)
