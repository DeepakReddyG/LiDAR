"""Experiment protocol/API checks using only synthetic fixtures and source code."""

import copy
import importlib.util
import itertools
import json
import runpy
import sys
import types
from pathlib import Path

import laspy
import numpy as np
import pytest

from revision import components, experiments


def test_registered_matrix_is_complete_unique_and_contains_no_extra_search():
    cases = experiments.configurations(True)
    assert len(cases) == len({row["id"] for row in cases}) == 28
    matched = [row for row in cases if row["id"].startswith(("rules_c", "sam3_c"))]
    observed = {
        (row["source"], row["constraints"], row["smoothing"], row["transfer"])
        for row in matched
    }
    expected = set(
        itertools.product(
            ["rules", "sam3"], [False, True], [False, True], ["naive", "height_aware"]
        )
    )
    assert observed == expected
    assert experiments.configurations(False) == [
        {
            "id": "sam3_c1_s1_height_aware",
            "source": "sam3",
            "constraints": True,
            "smoothing": True,
            "transfer": "height_aware",
        }
    ]


@pytest.fixture
def protocol(monkeypatch):
    bounds = {
        "allowed_context_bounds": [0, 0, 150, 150],
        "scoring_bounds": [10, 10, 50, 40],
    }
    guard = types.SimpleNamespace(
        _protocol=lambda: {"development": bounds},
        assert_development_bounds=lambda value: None,
    )
    monkeypatch.setattr(experiments.r, "guard_for", lambda m: guard)
    return {
        "dataset": {
            "context_bounds": bounds["allowed_context_bounds"].copy(),
            "score_bounds": bounds["scoring_bounds"].copy(),
            "input": "data/eval/tile_c.las",
            "reference": "data/eval/tile_c_gt.las",
        }
    }


def test_unsupported_input_is_rejected_before_open_or_hash(monkeypatch, protocol):
    protocol["dataset"]["input"] = "forbidden_input.las"
    monkeypatch.setattr(
        experiments.r,
        "sha",
        lambda path: pytest.fail("No data hash allowed before scope check"),
    )
    with pytest.raises(ValueError, match="preregistered tile C"):
        experiments.validate_scope(protocol)


@pytest.mark.parametrize("field", ["score_bounds", "context_bounds"])
def test_scope_must_match_locked_boxes(protocol, field):
    protocol["dataset"][field][0] += 1
    with pytest.raises(ValueError, match="bounds changed"):
        experiments.validate_scope(protocol)


def test_only_registered_one_factor_parameter_and_model_changes_are_accepted(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(experiments, "ROOT", tmp_path)
    path = tmp_path / "revision_work/manifests/base.json"
    path.parent.mkdir(parents=True)
    base = {
        **protocol,
        "parameters": {
            "GRID_RESOLUTION": 0.5,
            "TOP_SURFACE_FT": 1.5,
            "SOME_FIXED_VALUE": 8,
        },
        "model": {"checkpoint": "fixed"},
    }
    path.write_text(json.dumps(base))
    plan = tmp_path / "revision_work/experiment_plan.md"
    plan.write_text("fixed synthetic plan")
    manifest = copy.deepcopy(base)
    manifest.update(
        experiment_id="grid_025",
        base_manifest_path=path.relative_to(tmp_path).as_posix(),
        base_manifest_sha256=experiments.r.sha(path),
        registered_plan_sha256=experiments.r.sha(plan),
    )
    manifest["parameters"]["GRID_RESOLUTION"] = 0.25
    experiments.validate_registered_parameters(manifest)
    manifest["parameters"]["TOP_SURFACE_FT"] = 3
    with pytest.raises(ValueError, match="one-factor"):
        experiments.validate_registered_parameters(manifest)
    manifest["parameters"]["TOP_SURFACE_FT"] = 1.5
    manifest["model"]["checkpoint"] = "different"
    with pytest.raises(ValueError, match="Model settings"):
        experiments.validate_registered_parameters(manifest)


def test_manifest_creation_refuses_overwrite_before_data_open(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(experiments, "ROOT", tmp_path)
    folder = tmp_path / "revision_work/manifests"
    folder.mkdir(parents=True)
    base = folder / "base.json"
    base.write_text(json.dumps(protocol))
    existing = folder / "fixed_anchor.json"
    existing.write_text("preserve me")
    monkeypatch.setattr(
        experiments.laspy,
        "open",
        lambda *args, **kwargs: pytest.fail("Data must not be opened"),
    )
    with pytest.raises(FileExistsError, match="Preserve existing"):
        experiments.make_manifests(base)
    assert existing.read_text() == "preserve me"


@pytest.fixture
def matrix_fixture(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    values = runpy.run_path(str(root / "revision_work/source_snapshot/config.py"))
    parameters = {
        name: copy.deepcopy(value) for name, value in values.items() if name.isupper()
    }
    config = types.ModuleType("config")
    config.__dict__.update(parameters)
    monkeypatch.setitem(sys.modules, "config", config)
    spec = importlib.util.spec_from_file_location(
        "_experiment_export", root / "reprojection/map_back.py"
    )
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    parameters["CLASSES"] = {
        str(key): value for key, value in parameters["CLASSES"].items()
    }
    pilot = laspy.LasData(laspy.LasHeader(point_format=8, version="1.4"))
    pilot.x, pilot.y, pilot.z = [1, 1, 2, 2], [1, 1, 2, 2], [30, 0, 28, 0]
    pilot.add_extra_dim(laspy.ExtraBytesParams(name="orig_index", type=np.uint64))
    pilot.orig_index = [10, 11, 12, 13]
    reference = experiments.r.subset_las(pilot, np.ones(len(pilot.points), dtype=bool))
    reference.classification = [5, 3, 5, 3]
    data = {
        "pilot": pilot,
        "idx": np.arange(4),
        "x": np.asarray(pilot.x),
        "y": np.asarray(pilot.y),
        "z": np.asarray(pilot.z),
        "r": np.array([0, 0, 1, 1]),
        "c": np.array([0, 0, 1, 1]),
        "hag": np.asarray(pilot.z, np.float32),
        "exg": np.full(4, 0.2, np.float32),
        "surface": np.full((3, 3), 30, np.float32),
    }
    slices = tmp_path / "slices"
    slices.mkdir()
    np.save(slices / "exg_grid.npy", np.full((3, 3), 0.2, np.float32))
    np.save(slices / "hag_grid.npy", np.full((3, 3), 30, np.float32))
    np.save(slices / "void_mask.npy", np.zeros((3, 3), bool))
    manifest = {
        "parameters": parameters,
        "experiment_id": "fixed_anchor",
        "dataset": {
            "reference": "data/eval/tile_c_gt.las",
            "score_bounds": [0, 0, 4, 4],
        },
    }
    (tmp_path / "manifest.json").write_text("synthetic manifest identity")
    segment = types.SimpleNamespace(BBOX_FRAC_MAX=0.8)
    conf = {
        value["name"]: np.full(
            (3, 3), 0.9 if value["name"] == "tree" else 0, np.float32
        )
        for value in parameters["CLASSES"].values()
    }
    monkeypatch.setattr(experiments.r, "configure", lambda *args: config)
    monkeypatch.setattr(experiments, "contexts", lambda *args: data)
    monkeypatch.setattr(experiments.r, "load_raw_conf", lambda *args: conf)
    monkeypatch.setattr(
        experiments.r,
        "source_module",
        lambda m, name: segment if name == "segmentation.segment_sam3" else export,
    )
    events = {
        "prediction_count": 0,
        "required_before_reference": 1,
        "reference_reads": 0,
    }
    original_crosswalk = components.labels_to_las_codes

    def crosswalk(*args):
        events["prediction_count"] += 1
        return original_crosswalk(*args)

    monkeypatch.setattr(components, "labels_to_las_codes", crosswalk)

    def read_reference(path):
        assert str(path).endswith("data/eval/tile_c_gt.las")
        assert events["prediction_count"] == events["required_before_reference"]
        events["reference_reads"] += 1
        return reference

    monkeypatch.setattr(experiments.laspy, "read", read_reference)
    return tmp_path, manifest, events, segment, conf


def test_phase2_then_full_matrix_preserves_outputs_and_predicts_before_reference(
    matrix_fixture,
):
    out, manifest, events, _, _ = matrix_fixture
    experiments.matrix(manifest, out, full=False)
    original = (out / "analysis/metrics.json").read_bytes()
    assert set(json.loads(original)["methods"]) == {"sam3_c1_s1_height_aware"}
    events.update(prediction_count=0, required_before_reference=28)
    experiments.matrix(manifest, out, full=True)
    assert (out / "analysis/metrics.json").read_bytes() == original
    report = json.loads((out / "analysis_full/metrics.json").read_text())
    assert len(report["methods"]) == 28
    assert events["reference_reads"] == 2
    point = report["methods"]["point_rules"]
    assert point["routes"]["fallback"]["scored_points"] == 0
    assert point["routes"]["unknown"]["scored_points"] == 4
    assert all(
        row["integrity"]["geometry_checked"] for row in report["methods"].values()
    )
    with pytest.raises(FileExistsError):
        experiments.matrix(manifest, out, full=True)


def test_full_matrix_requires_completed_anchor_and_cannot_expand_sensitivity(tmp_path):
    with pytest.raises(ValueError, match="Phase 2 anchor"):
        experiments.matrix({"experiment_id": "fixed_anchor"}, tmp_path, full=True)
    with pytest.raises(ValueError, match="only at the fixed anchor"):
        experiments.matrix({"experiment_id": "grid_025"}, tmp_path, full=True)


def test_bbox_override_is_restored_on_response_failure(matrix_fixture, monkeypatch):
    out, manifest, _, segment, conf = matrix_fixture
    (out / "analysis").mkdir()
    (out / "analysis/metrics.json").write_text("completed synthetic Phase 2")

    def load(*args):
        if segment.BBOX_FRAC_MAX != 0.8:
            raise ValueError("Synthetic response failure")
        return conf

    monkeypatch.setattr(experiments.r, "load_raw_conf", load)
    with pytest.raises(ValueError, match="Synthetic response failure"):
        experiments.matrix(manifest, out, full=True)
    assert segment.BBOX_FRAC_MAX == 0.8
