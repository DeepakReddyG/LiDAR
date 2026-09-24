"""Matched component checks using synthetic arrays and the frozen code only."""

import copy
import importlib.util
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from revision.components import (
    classify_rules,
    fuse_components,
    geometry_only,
    labels_to_las_codes,
    majority_filter,
    rules_confidence,
    transfer_components,
)

SNAPSHOT = Path(__file__).resolve().parents[1] / "revision_work/source_snapshot"


@pytest.fixture
def parameters():
    config = runpy.run_path(str(SNAPSHOT / "config.py"))
    values = {
        name: copy.deepcopy(value) for name, value in config.items() if name.isupper()
    }
    values["CLASSES"] = {str(key): value for key, value in values["CLASSES"].items()}
    return values


def snapshot_module(relative, parameters, monkeypatch):
    # Only load source definitions, never call stages or access input/output data.
    config = types.ModuleType("config")
    config.__dict__.update(copy.deepcopy(parameters))
    config.CLASSES = {int(key): value for key, value in config.CLASSES.items()}
    monkeypatch.setitem(sys.modules, "config", config)
    spec = importlib.util.spec_from_file_location(
        "_component_reference", SNAPSHOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def confidence(parameters, shape, **values):
    names = [value["name"] for value in parameters["CLASSES"].values()]
    return {name: np.full(shape, values.get(name, 0), np.float32) for name in names}


@pytest.mark.parametrize(
    "height,physical", [(False, False), (True, False), (False, True), (True, True)]
)
def test_default_and_historical_variant_labels_scores_stats_match_snapshot(
    parameters, monkeypatch, height, physical
):
    inherited = snapshot_module("classification/fuse.py", parameters, monkeypatch)
    rng = np.random.default_rng(4321)
    shape = (15, 19)
    exg = rng.uniform(-0.2, 0.4, shape).astype(np.float32)
    hag = rng.uniform(-0.5, 15, shape).astype(np.float32)
    exg[3, 4] = np.nan
    hag[2, 1] = np.nan
    void = rng.random(shape) < 0.1
    conf = {
        value["name"]: rng.random(shape).astype(np.float32)
        for value in parameters["CLASSES"].values()
    }
    expected = inherited.fuse(
        conf, exg, hag, void, tree_height_check=height, physical_smoothing=physical
    )
    actual = fuse_components(
        conf,
        exg,
        hag,
        void,
        parameters,
        tree_height_check=height,
        physical_smoothing=physical,
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_constraints_toggle_changes_only_height_color_checks_with_fixed_void(
    parameters,
):
    shape = (1, 2)
    conf = confidence(parameters, shape, tree=0.9)
    void = np.array([[False, True]])
    zeros = np.zeros(shape, np.float32)
    enabled, _, _ = fuse_components(
        conf, zeros, zeros, void, parameters, smoothing=False
    )
    disabled, _, stats = fuse_components(
        conf, zeros, zeros, void, parameters, constraints=False, smoothing=False
    )
    assert enabled.tolist() == [[-1, -1]]
    assert disabled.tolist() == [[4, -1]]
    assert stats["tree"]["claimed_px"] == 2
    assert stats["tree"]["kept_px"] == 1


def test_smoothing_toggle_and_window_isolate_propagation(parameters):
    shape = (5, 5)
    conf = confidence(parameters, shape, tree=0.9)
    conf["tree"][2, 2] = 0
    conf["grass"][2, 2] = 0.9
    hag, exg, void = np.zeros(shape), np.full(shape, 0.2), np.zeros(shape, bool)
    untouched, _, _ = fuse_components(
        conf, exg, hag, void, parameters, constraints=False, smoothing=False
    )
    smoothed, _, _ = fuse_components(
        conf, exg, hag, void, parameters, constraints=False, majority_size=3
    )
    size_one, _, _ = fuse_components(
        conf, exg, hag, void, parameters, constraints=False, majority_size=1
    )
    assert untouched[2, 2] == 3
    assert smoothed[2, 2] == 4
    np.testing.assert_array_equal(size_one, untouched)


def test_majority_preserves_inherited_tie_and_edge_rules(parameters, monkeypatch):
    inherited = snapshot_module("classification/fuse.py", parameters, monkeypatch)
    grid = np.array([[-1, 0, 1], [2, -1, 0], [1, 2, -1]], dtype=np.int32)
    for size in [1, 2, 3, 5, 9]:
        np.testing.assert_array_equal(
            majority_filter(grid, size), inherited.majority_filter(grid, size)
        )
    with pytest.raises(ValueError, match="positive integer"):
        majority_filter(grid, 0)


def test_rule_proposals_and_pointwise_labels_match_baseline(parameters, monkeypatch):
    inherited = snapshot_module("classification/baseline.py", parameters, monkeypatch)
    exg = np.array([0, 0.2, 0.2, 0, 0.05, 0.2, np.nan], np.float32)
    hag = np.array([10, 10, 0.2, 0.2, 2, 6, 0], np.float32)
    labels = classify_rules(exg, hag, parameters)
    np.testing.assert_array_equal(labels, inherited.classify(exg, hag))
    assert labels.tolist() == [5, 4, 3, 0, -1, -1, -1]
    proposals = rules_confidence(exg.reshape(1, -1), hag.reshape(1, -1), parameters)
    assert len(proposals) == 7
    np.testing.assert_array_equal(sum(proposals.values()), (labels >= 0).reshape(1, -1))
    assert not proposals["vehicle"].any()
    assert not proposals["sidewalk"].any()
    assert not proposals["parking"].any()


def test_geometry_proxy_is_explicitly_restricted_and_strict_at_thresholds(parameters):
    hag = np.array([-0.5, 0, 1.99, 2, 6, 6.01, np.nan, np.inf])
    assert geometry_only(hag, parameters).tolist() == [3, 3, 3, -1, -1, 4, -1, -1]


def test_transfer_default_labels_match_snapshot_and_split_assignment_sources(
    parameters, monkeypatch
):
    inherited = snapshot_module("reprojection/map_back.py", parameters, monkeypatch)
    grid = np.array([[4, -1, 3, 0]], np.int32)
    score = np.array([[0.8, 0, 0.7, 0.4]], np.float32)
    surface = np.array([[30, 30, np.nan, 30]], np.float32)
    r = np.zeros(8, np.int32)
    c = np.array([0, 0, 0, 0, 1, 2, 3, 0], np.int32)
    z = np.array([29, 15, 0.5, 0.5, 29, 0, 35, 27], np.float32)
    hag = np.array([29, 15, 0.5, 0.5, 29, 0, 1, 27], np.float32)
    exg = np.array([0.3, 0.1, 0.2, 0, 0.1, 0.2, 0, 0.1], np.float32)
    expected, _ = inherited.label_points(z, r, c, grid, score, surface, hag, exg)
    actual = transfer_components(z, r, c, grid, score, surface, hag, exg, parameters)
    np.testing.assert_array_equal(actual["labels"], expected)
    assert actual["prediction_source"].tolist() == [1, 2, 2, 2, 0, 3, 4, 2]
    assert actual["model_score"][0] == np.float32(0.8)
    assert np.isnan(actual["model_score"][1:]).all()
    assert actual["routes"].tolist() == [
        "transfer",
        "fallback",
        "fallback",
        "fallback",
        "transfer",
        "fallback",
        "fallback",
        "fallback",
    ]


def test_naive_mode_directly_transfers_all_heights_and_retains_abstention(parameters):
    grid = np.array([[4, -1]], np.int32)
    score = np.array([[0.8, 0]], np.float32)
    surface = np.array([[30, np.nan]], np.float32)
    result = transfer_components(
        [0, 0],
        np.array([0, 0]),
        np.array([0, 1]),
        grid,
        score,
        surface,
        [0, 0],
        [0.2, 0.2],
        parameters,
        mode="naive",
    )
    assert result["labels"].tolist() == [4, -1]
    assert result["prediction_source"].tolist() == [1, 0]
    assert result["routes"].tolist() == ["transfer", "transfer"]
    assert np.isnan(result["model_score"][1])


def test_transfer_band_sensitivity_changes_only_surface_membership(parameters):
    args = (
        [28],
        np.array([0]),
        np.array([0]),
        np.array([[4]], np.int32),
        np.array([[0.8]], np.float32),
        np.array([[30]], np.float32),
        [0],
        [0.2],
        parameters,
    )
    narrow = transfer_components(*args, band=1.5)
    wide = transfer_components(*args, band=3.0)
    assert narrow["labels"].tolist() == [3]
    assert wide["labels"].tolist() == [4]
    with pytest.raises(ValueError, match="finite and positive"):
        transfer_components(*args, band=0)


def test_explicit_las_crosswalk_rejects_unknown_internal_id(parameters):
    assert labels_to_las_codes(
        np.array([-1, 0, 1, 2, 3, 4, 5, 6]), parameters
    ).tolist() == [1, 11, 11, 11, 3, 5, 6, 64]
    with pytest.raises(ValueError, match="Unsupported internal"):
        labels_to_las_codes(np.array([7]), parameters)


def test_inputs_are_not_mutated(parameters):
    exg = np.full((3, 3), 0.2, np.float32)
    hag = np.full((3, 3), 10, np.float32)
    void = np.zeros((3, 3), bool)
    conf = confidence(parameters, (3, 3), tree=0.9)
    for array in [exg, hag, void, *conf.values()]:
        array.flags.writeable = False
    labels, _, _ = fuse_components(conf, exg, hag, void, parameters)
    assert np.all(labels == 4)
    assert np.all(conf["tree"] == np.float32(0.9))


def test_invalid_proposals_and_out_of_bounds_transfer_fail(parameters):
    conf = confidence(parameters, (2, 2), tree=1.1)
    with pytest.raises(ValueError, match=r"finite in \[0,1\]"):
        fuse_components(
            conf, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2), bool), parameters
        )
    with pytest.raises(ValueError, match="outside the grid"):
        transfer_components(
            [0],
            np.array([-1]),
            np.array([0]),
            np.array([[4]], np.int32),
            np.ones((1, 1)),
            np.ones((1, 1)),
            [0],
            [0],
            parameters,
        )
