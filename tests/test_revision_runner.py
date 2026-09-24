"""Synthetic runner preflight checks; never inspect real or held-out point data."""

import json
import subprocess
import sys
import types

import laspy
import numpy as np
import pytest
from PIL import Image

from revision import runner


def test_geometry_checks_header_scaling_and_order_and_coordinates():
    reference = laspy.LasData(laspy.LasHeader(point_format=8, version="1.4"))
    reference.x = [1, 2]
    reference.y = [3, 4]
    reference.z = [5, 6]
    reference.add_extra_dim(laspy.ExtraBytesParams(name="orig_index", type=np.uint64))
    reference.orig_index = [10, 11]
    selected = np.array([True, True])

    def clone():
        result = laspy.LasData(reference.header.copy())
        result.points = reference.points.copy()
        return result

    for field in ("scales", "offsets", "X", "orig_index"):
        prediction = clone()
        if field in ("scales", "offsets"):
            setattr(prediction.header, field, getattr(prediction.header, field) + 1)
        else:
            setattr(prediction, field, np.asarray(getattr(prediction, field)) + 1)
        with pytest.raises(ValueError, match="changed|ordering"):
            runner.assert_geometry(prediction, reference, selected)
    runner.assert_geometry(clone(), reference, selected)


def test_boundary_is_3d_euclidean_excludes_unreviewed_and_includes_perimeter():
    xyz = np.array(
        [
            [10, 10, 0],
            [10, 10, 20],  # Vertical overlap is not a boundary.
            [15, 15, 0],
            [15.6, 15.6, 0],  # 0.85 ft, cross-class boundary.
            [20, 20, 0],
            [20.9, 20.9, 0],  # Square-near but 1.27 ft away.
            [25, 25, 0],
            [25.1, 25, 0],  # Nearby unknown cannot create a boundary.
            [0.5, 10, 0],  # True crop perimeter.
        ]
    )
    codes = np.array([3, 5, 3, 5, 3, 5, 3, 1, 3])
    result = runner.reference_boundary(*xyz.T, codes, [0, 0, 40, 30])
    assert result.tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
    ]


def test_missing_surface_is_not_reported_as_below_surface():
    result = runner.surface_strata(
        np.array([10, 6, 14, 0]), np.array([10, 10, 10, np.nan]), 3
    )
    assert result["surface"].tolist() == [True, False, False, False]
    assert result["below_surface"].tolist() == [False, True, False, False]
    assert result["above_surface"].tolist() == [False, False, True, False]
    assert result["missing_surface"].tolist() == [False, False, False, True]


@pytest.fixture
def validation_manifest(monkeypatch):
    locked = {
        "allowed_context_bounds": [0, 0, 150, 150],
        "scoring_bounds": [10, 10, 50, 40],
    }
    guard = types.SimpleNamespace(
        assert_development_bounds=lambda bounds: None,
        _protocol=lambda: {"development": locked},
    )
    monkeypatch.setattr(runner, "guard_for", lambda m: guard)
    monkeypatch.setattr(
        runner,
        "actual_csf_parameters",
        lambda resolution: {"cloth_resolution": resolution},
    )
    monkeypatch.setattr(runner, "sha", lambda path: "hash")
    return {
        "source_root": "revision_work/source_snapshot",
        "dataset": {
            "context_bounds": locked["allowed_context_bounds"].copy(),
            "score_bounds": locked["scoring_bounds"].copy(),
            "input": "data/eval/tile_c.las",
            "reference": "data/eval/tile_c_gt.las",
            "input_sha256": "hash",
            "reference_sha256": "hash",
        },
        "variant": runner.DEFAULT_VARIANT.copy(),
        "parameters": {"CLOTH_RESOLUTION": 2},
        "csf_parameters": {"cloth_resolution": 2},
        "source_hashes": {},
    }


@pytest.mark.parametrize("field", ["context_bounds", "score_bounds"])
def test_smaller_context_or_scoring_scope_cannot_change_denominator(
    validation_manifest, field
):
    validation_manifest["dataset"][field][2] -= 1
    with pytest.raises(ValueError, match="exactly match preregistration"):
        runner.validate(validation_manifest)


def test_unsupported_variant_and_changed_csf_defaults_fail(validation_manifest):
    runner.validate(validation_manifest)
    validation_manifest["variant"]["smoothing"] = False
    with pytest.raises(ValueError, match="default variant"):
        runner.validate(validation_manifest)
    validation_manifest["variant"] = runner.DEFAULT_VARIANT.copy()
    validation_manifest["csf_parameters"]["cloth_resolution"] = 3
    with pytest.raises(ValueError, match="CSF parameters"):
        runner.validate(validation_manifest)


def test_score_refuses_existing_outputs_before_reading_inputs(tmp_path):
    (tmp_path / "output").mkdir()
    sentinel = tmp_path / "output/metrics.json"
    sentinel.write_text("preserved")
    with pytest.raises(FileExistsError):
        runner.score({}, tmp_path)
    assert sentinel.read_text() == "preserved"


def test_imported_module_must_be_inside_snapshot(tmp_path):
    outside = types.SimpleNamespace(
        __name__="config", __file__=str(tmp_path / "elsewhere/config.py")
    )
    with pytest.raises(RuntimeError, match="outside the frozen source"):
        runner.assert_source_module(outside, tmp_path / "snapshot")


def test_configuration_cannot_change_in_one_process(tmp_path):
    source = tmp_path / "revision_work/source_snapshot"
    source.mkdir(parents=True)
    (source / "config.py").write_text("GRID_RESOLUTION=0.5\n")
    manifest = {
        "source_root": "revision_work/source_snapshot",
        "source_hashes": {
            "revision_work/source_snapshot/config.py": runner.sha(source / "config.py")
        },
        "parameters": {"GRID_RESOLUTION": 0.5},
    }
    script = """
import json, sys
from pathlib import Path
from revision import runner
runner.ROOT=Path(sys.argv[1])
m=json.loads(sys.argv[2]);out=runner.ROOT/'output'
runner.configure(m,out)
runner.configure(m,out)
m['parameters']['GRID_RESOLUTION']=1
try:
    runner.configure(m,out)
except RuntimeError as e:
    assert 'one pipeline configuration' in str(e)
else:
    raise AssertionError('Configuration mutation was accepted')
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), json.dumps(manifest)], check=True
    )


@pytest.fixture
def response_fixture(tmp_path):
    image = tmp_path / "slices/tiles/tile_r0_c0.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (4, 3), "green").save(image)
    output = tmp_path / "masks/raw/tile_r0_c0.json"
    source_hashes = {
        runner.BPE_PATH: "bpe",
        "revision/sam_worker.py": "worker",
        "segmentation/mlx_sam3/sam3/model.py": "model",
    }
    manifest = {
        "model": {
            "sha256": "checkpoint",
            "checkpoint": str(tmp_path / "checkpoint.safetensors"),
            "raw_score_floor": 0.1,
            "processor_resolution": 1008,
            "seed": 0,
        },
        "parameters": {"CLASSES": {"3": {"prompt": "grass"}, "4": {"prompt": "tree"}}},
        "source_hashes": source_hashes,
    }
    runner.dump(tmp_path / "manifest.json", manifest)
    runner.dump(
        tmp_path / "sam_jobs.json",
        [
            {
                "image": str(image),
                "output": str(output),
                "image_sha256": runner.sha(image),
            }
        ],
    )
    response = {
        "image_path": str(image),
        "image_sha256": runner.sha(image),
        "width": 4,
        "height": 3,
        "results": {"grass": [], "tree": []},
        "run_manifest": {
            "parent_manifest_sha256": runner.sha(tmp_path / "manifest.json"),
            "checkpoint_sha256": "checkpoint",
            "checkpoint_path": manifest["model"]["checkpoint"],
            "source_sha256": source_hashes.copy(),
            "prompts": ["grass", "tree"],
            "raw_score_floor": 0.1,
            "seed": 0,
            "preprocessing": {"resize": [1008, 1008]},
        },
    }
    runner.dump(output, response)
    return tmp_path, manifest, output, response


def test_complete_response_identity_is_accepted(response_fixture):
    root, manifest, _, _ = response_fixture
    assert len(runner.verified_responses(manifest, root)) == 1


@pytest.mark.parametrize(
    "change,match",
    [
        ("prompt", "prompts"),
        ("checkpoint", "checkpoint"),
        ("code", "code identity"),
        ("image", "image identity"),
        ("parent", "parent manifest"),
        ("resolution", "resolution"),
        ("missing", "missing or unexpected"),
        ("extra", "missing or unexpected"),
    ],
)
def test_partial_or_stale_responses_cannot_be_scored(response_fixture, change, match):
    root, manifest, output, response = response_fixture
    if change == "prompt":
        del response["results"]["tree"]
    elif change == "checkpoint":
        response["run_manifest"]["checkpoint_sha256"] = "different"
    elif change == "code":
        response["run_manifest"]["source_sha256"]["revision/sam_worker.py"] = (
            "different"
        )
    elif change == "image":
        response["image_sha256"] = "different"
    elif change == "parent":
        response["run_manifest"]["parent_manifest_sha256"] = "different"
    elif change == "resolution":
        response["run_manifest"]["preprocessing"]["resize"] = [512, 512]
    runner.dump(output, response)
    if change == "missing":
        output.unlink()
    elif change == "extra":
        runner.dump(output.parent / "tile_r1_c1.json", response)
    with pytest.raises(ValueError, match=match):
        runner.verified_responses(manifest, root)


def test_csf_manifest_contains_actual_runtime_parameters_without_filtering():
    parameters = runner.actual_csf_parameters(2.0)
    assert parameters["cloth_resolution"] == 2.0
    assert parameters["bSloopSmooth"] is False
    assert set(parameters) == {
        "bSloopSmooth",
        "class_threshold",
        "cloth_resolution",
        "interations",
        "rigidness",
        "time_step",
    }
    json.dumps(parameters, allow_nan=False)
