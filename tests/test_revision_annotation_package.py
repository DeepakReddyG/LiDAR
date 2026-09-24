"""All data, protocols, gates and repositories in this file are synthetic."""

import json
import subprocess

import laspy
import numpy as np
import pytest

from revision.annotation_package import (
    _audit_sample,
    export_holdout_packages,
    export_pilot_audit,
)
from revision.guard import (
    HoldoutGuard,
    Phase5AlreadyConsumed,
    ProtocolViolation,
    begin_phase5,
    finish_phase5,
    sha256_file,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True).stdout.decode().strip()


def write_cloud(path, bounds, *, original_offset=100_000_000):
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0, 0, 700]
    cloud = laspy.LasData(header)
    xx, yy = np.meshgrid(np.arange(bounds[0] + 0.25, bounds[2], 0.5),
                         np.arange(bounds[1] + 0.25, bounds[3], 0.5))
    cloud.x, cloud.y = xx.ravel(), yy.ravel()
    n = len(cloud.points)
    cloud.z = 800 + (np.arange(n) % 3) * 5
    cloud.red = np.full(n, 20000, np.uint16)
    cloud.green = np.where(np.arange(n) % 2, 30000, 19000).astype(np.uint16)
    cloud.blue = np.full(n, 20000, np.uint16)
    cloud.intensity = np.arange(n, dtype=np.uint16) + 123
    cloud.return_number = np.ones(n, dtype=np.uint8)
    cloud.number_of_returns = np.ones(n, dtype=np.uint8)
    cloud.classification = np.where(np.arange(n) % 2, 3, 5).astype(np.uint8)
    cloud.user_data = np.full(n, 88, np.uint8)
    cloud.classification_flags = np.full(n, 3, np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="orig_index", type=np.uint64))
    cloud.orig_index = np.arange(n, dtype=np.uint64) + original_offset
    for name in ["prediction", "reference", "confidence", "fallback_source"]:
        cloud.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
        cloud[name] = np.full(n, 0.99, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.write(path)
    return cloud


@pytest.fixture
def setup(tmp_path):
    protocol = {
        "schema_version": 1,
        "regions": [{"id": "holdout_a", "source_candidate": "data/eval/tile_a_gt.las",
                     "protected_bounds": [100, 0, 110, 10], "scoring_bounds": [102, 2, 104, 4]},
                    {"id": "holdout_b", "source_candidate": "data/eval/tile_b_gt.las",
                     "protected_bounds": [200, 0, 210, 10], "scoring_bounds": [202, 2, 204, 4]}],
        "development": {"allowed_context_bounds": [0, 0, 10, 10], "scoring_bounds": [2, 2, 8, 8]},
    }
    write_json(tmp_path / "holdout.json", protocol)
    (tmp_path / "objective.md").write_text("Synthetic immutable equal-cost objective\n")
    guard = HoldoutGuard.from_files(tmp_path / "holdout.json", tmp_path / "objective.md",
                                   expected_holdout_sha256=sha256_file(tmp_path / "holdout.json"),
                                   expected_objective_sha256=sha256_file(tmp_path / "objective.md"))
    raw = tmp_path / "data/eval/tile_c.las"
    cloud = write_cloud(raw, protocol["development"]["scoring_bounds"])
    arrays_path = tmp_path / "revision_work/runs/phase1/output/pilot_arrays.npz"
    arrays_path.parent.mkdir(parents=True)
    arrays = {
        "ids": np.asarray(cloud.orig_index), "x": np.asarray(cloud.x),
        "y": np.asarray(cloud.y), "z": np.asarray(cloud.z),
        "rgb": np.column_stack([cloud.red, cloud.green, cloud.blue]),
        "hag": np.arange(len(cloud.points)) % 10,
        "reference": np.asarray(cloud.classification),
        "prediction": np.zeros(len(cloud.points), dtype=np.uint8),
        "boundary": np.ones(len(cloud.points), dtype=bool),
    }
    np.savez_compressed(arrays_path, **arrays)
    manifest = {"phase": 1, "dataset": {"input": "data/eval/tile_c.las", "input_sha256": sha256_file(raw)},
                "protocol": {"holdout_sha256": guard.expected_holdout_sha256,
                             "objective_sha256": guard.expected_objective_sha256}}
    write_json(arrays_path.parent.parent / "manifest.json", manifest)
    write_json(arrays_path.parent / "artifact_hashes.json", {"output/pilot_arrays.npz": sha256_file(arrays_path)})
    return {"root": tmp_path, "guard": guard, "raw": raw, "arrays_path": arrays_path,
            "arrays": arrays, "protocol": protocol}


def run_pilot(setup, output="audit"):
    return export_pilot_audit(arrays_path=setup["arrays_path"], raw_tile_path=setup["raw"],
                              output_dir=setup["root"] / output, guard=setup["guard"])


def test_pilot_audit_hides_predictions_and_original_references(setup):
    before = sha256_file(setup["raw"])
    result = run_pilot(setup)
    assert result["target_count"] <= 1200 and not result["labels_created"]
    public = setup["root"] / "audit/annotator/pilot_c_audit"
    cloud = laspy.read(public / "annotation_input.las")
    assert set(cloud.point_format.extra_dimension_names) == {"tile_index", "audit_target"}
    assert not np.any(cloud.classification)
    assert not np.any(cloud.classification_flags)
    assert not np.any(cloud.user_data)
    assert np.array_equal(cloud.tile_index, np.arange(len(cloud.points)))
    for dim in ["x", "y", "z"]:
        assert np.array_equal(np.asarray(getattr(cloud, dim)), setup["arrays"][dim])
    assert np.array_equal(np.column_stack([cloud.red, cloud.green, cloud.blue]), setup["arrays"]["rgb"])
    assert sha256_file(setup["raw"]) == before
    assert not list((setup["root"] / "data/eval").glob("*_gt.las"))
    with np.load(setup["root"] / "audit/private/pilot_c_audit/identity_map.npz") as mapping:
        assert np.array_equal(mapping["original_id"], setup["arrays"]["ids"])
        assert set(mapping.files) == {"local_id", "original_id", "source_row", "source_sha256"}
    response = (public / "response_template.csv").read_text().splitlines()
    assert len(response) == result["target_count"] + 1
    assert all(line.endswith(",,,,,") for line in response[1:])


def test_pilot_refuses_overwrite(setup):
    run_pilot(setup)
    with pytest.raises(FileExistsError):
        run_pilot(setup)


def test_audit_targets_ignore_prediction_and_original_class_values(setup):
    arrays = setup["arrays"]
    bounds = setup["protocol"]["development"]["allowed_context_bounds"]
    first, _ = _audit_sample(arrays, bounds)
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["prediction"][:] = 5
    changed["reference"][:] = 6  # membership unchanged, semantic class differs
    changed["boundary"][:] = False
    second, _ = _audit_sample(changed, bounds)
    assert np.array_equal(first, second)
    order = np.arange(len(arrays["ids"]))[::-1]
    reordered = {k: v[order] for k, v in arrays.items()}
    third, _ = _audit_sample(reordered, bounds)
    assert np.array_equal(first, third)


def test_no_more_than_fifty_per_stratum_and_weighted_population_counts():
    n = 5000
    arrays = {"ids": np.arange(n, dtype=np.uint64), "x": np.full(n, 0.25),
              "y": np.full(n, 0.25), "z": np.ones(n), "hag": np.ones(n),
              "rgb": np.tile([100, 120, 100], (n, 1)), "reference": np.full(n, 3)}
    targets, record = _audit_sample(arrays, [0, 0, 2, 2])
    assert len(targets) == 50
    assert sum(s["N_h"] for s in record["strata"]) == n
    assert max(s["n_h"] for s in record["strata"]) == 50
    populated = [s for s in record["strata"] if s["N_h"]]
    assert populated[0]["inclusion_probability"] == 0.01


def test_pilot_wrong_input_is_rejected_before_open(setup, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No LAS access allowed")
    monkeypatch.setattr(laspy, "open", forbidden)
    with pytest.raises(ProtocolViolation, match="only the fixed raw tile C"):
        export_pilot_audit(arrays_path=setup["arrays_path"],
            raw_tile_path=setup["root"] / "data/eval/tile_a.las",
            output_dir=setup["root"] / "wrong", guard=setup["guard"])


def test_source_and_array_hash_mismatch_fails(setup):
    with setup["arrays_path"].open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ProtocolViolation, match="artifact hash"):
        run_pilot(setup)


def test_coordinate_mismatch_is_detected_without_modifying_source(setup):
    before = sha256_file(setup["raw"])
    arrays = {k: v.copy() for k, v in setup["arrays"].items()}
    arrays["z"][0] += 0.01
    np.savez_compressed(setup["arrays_path"], **arrays)
    write_json(setup["arrays_path"].parent / "artifact_hashes.json",
               {"output/pilot_arrays.npz": sha256_file(setup["arrays_path"])})
    with pytest.raises(ProtocolViolation, match="coordinate correspondence"):
        run_pilot(setup)
    assert sha256_file(setup["raw"]) == before


@pytest.fixture
def frozen(setup):
    root = setup["root"]
    for letter, offset in [("a", 100), ("b", 200)]:
        write_cloud(root / f"data/eval/tile_{letter}.las", [offset, 0, offset + 10, 10])
    manifest = root / "final_manifest.json"
    write_json(manifest, {"kind": "synthetic final manifest"})
    git(root, "init", "-q")
    git(root, "add", "holdout.json", "objective.md", "final_manifest.json")
    git(root, "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "Synthetic final freeze")
    gate = root / "phase5.json"
    event = begin_phase5(gate, guard=setup["guard"], manifest_path=manifest,
                         repository=root, final_commit=git(root, "rev-parse", "HEAD"), mode="annotation_only")
    return {"repository": root, "gate_path": gate, "manifest_path": manifest,
            "output_dir": root / "holdout_package", "guard": setup["guard"], "event_id": event["event_id"]}


def test_holdout_requires_matching_gate_before_any_las_read(frozen, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No LAS access allowed")
    monkeypatch.setattr(laspy, "open", forbidden)
    with pytest.raises(ProtocolViolation, match="does not match"):
        export_holdout_packages(**{**frozen, "event_id": "wrong"})
    assert not frozen["output_dir"].exists()


def test_holdout_exports_fixed_boxes_once_without_references(frozen):
    before = {str(p): sha256_file(p) for p in (frozen["repository"] / "data/eval").glob("*.las")}
    event_bytes = frozen["gate_path"].read_bytes()
    result = export_holdout_packages(**frozen)
    assert result["accuracy"] is None and not result["labels_created"]
    assert [r["point_count"] for r in result["regions"]] == [16, 16]
    assert frozen["gate_path"].read_bytes() == event_bytes
    for name, expected_x in [("holdout_a", 102), ("holdout_b", 202)]:
        cloud = laspy.read(frozen["output_dir"] / "annotator" / name / "annotation_input.las")
        assert set(cloud.point_format.extra_dimension_names) == {"tile_index"}
        assert not np.any(cloud.classification)
        assert np.min(cloud.x) >= expected_x and np.max(cloud.x) < expected_x + 2
    after = {str(p): sha256_file(p) for p in (frozen["repository"] / "data/eval").glob("*.las")}
    assert before == after
    assert not list((frozen["repository"] / "data/eval").glob("*_gt.las"))
    with pytest.raises(Phase5AlreadyConsumed):
        export_holdout_packages(**{**frozen, "output_dir": frozen["repository"] / "another_export"})


def test_finished_or_changed_manifest_gate_cannot_export(frozen):
    finish_phase5(frozen["gate_path"], event_id=frozen["event_id"], status="blocked_no_reference")
    with pytest.raises(ProtocolViolation, match="already completed"):
        export_holdout_packages(**frozen)


def test_manifest_must_match_gate_before_sources_open(frozen, monkeypatch):
    frozen["manifest_path"].write_text('{"changed": true}')
    def forbidden(*args, **kwargs):
        raise AssertionError("No LAS access allowed")
    monkeypatch.setattr(laspy, "open", forbidden)
    with pytest.raises(ProtocolViolation, match="frozen bytes"):
        export_holdout_packages(**frozen)


def test_partial_export_claim_is_not_retryable(frozen):
    # Simulated previous crash reserves an empty sidecar. No raw source access.
    claim = frozen["gate_path"].with_name(frozen["gate_path"].name + ".annotation_export.json")
    claim.touch()
    with pytest.raises(Phase5AlreadyConsumed):
        export_holdout_packages(**frozen)
