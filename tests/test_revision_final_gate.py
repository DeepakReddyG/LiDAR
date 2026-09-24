"""Synthetic-only Phase 5 preparation tests; no actual gate or A/B file read."""

import json
import shutil
import subprocess
from pathlib import Path

import laspy
import numpy as np
import pytest

from revision.final_gate import run_final_gate
from revision.guard import Phase5AlreadyConsumed, ProtocolViolation, sha256_file


def git(root, *args):
    return (
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )


def cloud(path, code):
    las = laspy.LasData(laspy.LasHeader(point_format=8, version="1.4"))
    las.x = [100.25, 100.75]
    las.y = [0.25, 0.75]
    las.z = [10.0, 12.0]
    las.classification = [code, code]
    las.red = [1000, 2000]
    las.green = [2000, 3000]
    las.blue = [1000, 2000]
    las.add_extra_dim(laspy.ExtraBytesParams(name="orig_index", type=np.uint64))
    las.orig_index = [100_000_000, 100_000_001]
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)


@pytest.fixture
def frozen(tmp_path):
    lock = {
        "schema_version": 1,
        "regions": [
            {
                "id": "holdout_a",
                "source_candidate": "data/eval/tile_a_gt.las",
                "protected_bounds": [100, 0, 110, 10],
                "scoring_bounds": [100, 0, 101, 1],
            }
        ],
        "development": {
            "allowed_context_bounds": [0, 0, 10, 10],
            "scoring_bounds": [2, 2, 8, 8],
        },
    }
    (tmp_path / "holdout.json").write_text(json.dumps(lock))
    (tmp_path / "objective.md").write_text("Synthetic immutable objective\n")
    manifest = {
        "protocol": {
            "holdout_sha256": sha256_file(tmp_path / "holdout.json"),
            "objective_sha256": sha256_file(tmp_path / "objective.md"),
        },
        "phase5": {
            "gate_path": "revision_work/phase5_gate.json",
            "annotation_output": "documents/holdout_annotation",
            "approved_reference_records": [],
        },
    }
    project = Path(__file__).resolve().parents[1]
    (tmp_path / "revision").mkdir()
    source_names = [
        "revision/final_gate.py",
        "revision/annotation_package.py",
        "revision/guard.py",
    ]
    for name in source_names:
        shutil.copyfile(project / name, tmp_path / name)
    manifest["source_hashes"] = {
        name: sha256_file(tmp_path / name) for name in source_names
    }
    manifest_path = tmp_path / "final_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    git(tmp_path, "init", "-q")
    git(
        tmp_path,
        "add",
        "holdout.json",
        "objective.md",
        "final_manifest.json",
        *source_names,
    )
    git(
        tmp_path,
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=synthetic@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Synthetic final freeze",
    )
    cloud(tmp_path / "data/eval/tile_a.las", 0)
    return {
        "repository": tmp_path,
        "manifest_path": manifest_path,
        "final_commit": git(tmp_path, "rev-parse", "HEAD"),
    }


def test_missing_references_exports_once_and_stops_without_score(frozen):
    result = run_final_gate(**frozen)
    assert result["status"] == "blocked_no_reference"
    assert result["heldout_accuracy"] is None and not result["model_inference_run"]
    root = frozen["repository"]
    inventory = json.loads(
        (root / "revision_work/phase5_gate.json.reference_inventory.json").read_text()
    )
    assert inventory["records"][0]["exists"] is False
    annotation = laspy.read(
        root / "documents/holdout_annotation/annotator/holdout_a/annotation_input.las"
    )
    assert len(annotation.points) == 2 and not np.any(annotation.classification)
    completion = json.loads(
        (root / "revision_work/phase5_gate.json.completion.json").read_text()
    )
    assert completion["status"] == "blocked_no_reference"
    with pytest.raises((FileExistsError, Phase5AlreadyConsumed)):
        run_final_gate(**frozen)


def test_nonzero_candidate_classes_are_not_accepted_as_approved(frozen):
    source = frozen["repository"] / "data/eval/tile_a_gt.las"
    cloud(source, 5)
    before = sha256_file(source)
    result = run_final_gate(**frozen)
    inventory = json.loads(Path(result["reference_inventory"]).read_text())["records"][
        0
    ]
    assert inventory["non_placeholder_classification_count"] == 2
    assert inventory["candidate_classification_counts"] == {"5": 2}
    assert not inventory["eligible_for_scoring"]
    assert "UNKNOWN" in inventory["annotation_provenance"]
    assert sha256_file(source) == before
    assert (
        result["status"] == "blocked_no_reference"
        and result["heldout_accuracy"] is None
    )


def test_invalid_final_commit_stops_before_any_candidate_or_raw_access(
    frozen, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("No LAS access before gate")

    monkeypatch.setattr(laspy, "open", forbidden)
    with pytest.raises(ProtocolViolation, match="HEAD"):
        run_final_gate(**{**frozen, "final_commit": "0" * 40})
    assert not (frozen["repository"] / "revision_work/phase5_gate.json").exists()


def test_gate_and_output_paths_cannot_be_changed_to_bypass_one_shot(frozen):
    with pytest.raises(ProtocolViolation, match="Gate path differs"):
        run_final_gate(**frozen, gate_path=frozen["repository"] / "another_gate.json")
    with pytest.raises(ProtocolViolation, match="Annotation output differs"):
        run_final_gate(**frozen, output_dir=frozen["repository"] / "another_package")


def test_failure_after_consumption_is_terminal_not_retryable(frozen):
    # An invalid candidate is still never silently used as reference. It causes
    # a failed consumed event, not an automatic second try.
    source = frozen["repository"] / "data/eval/tile_a_gt.las"
    source.write_bytes(b"synthetic invalid LAS")
    with pytest.raises(laspy.errors.LaspyException):
        run_final_gate(**frozen)
    completion = json.loads(
        (
            frozen["repository"] / "revision_work/phase5_gate.json.completion.json"
        ).read_text()
    )
    assert completion["status"] == "failed"
    with pytest.raises(Phase5AlreadyConsumed):
        run_final_gate(**frozen)


def test_nonempty_approval_records_require_a_different_verified_implementation(
    frozen, monkeypatch
):
    manifest = json.loads(frozen["manifest_path"].read_text())
    manifest["phase5"]["approved_reference_records"] = [
        {"claimed_approval": "unverified"}
    ]
    frozen["manifest_path"].write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        raise AssertionError("No LAS access before gate")

    monkeypatch.setattr(laspy, "open", forbidden)
    with pytest.raises(
        ProtocolViolation, match="Nonempty approval records require verification"
    ):
        run_final_gate(**frozen)
    assert not (frozen["repository"] / "revision_work/phase5_gate.json").exists()


def test_modified_executable_source_blocks_access_before_consuming_gate(
    frozen, monkeypatch
):
    source = frozen["repository"] / "revision/annotation_package.py"
    source.write_text(source.read_text() + "\n# changed after freeze\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("No LAS access before gate")

    monkeypatch.setattr(laspy, "open", forbidden)
    with pytest.raises(ProtocolViolation, match="executable source differs"):
        run_final_gate(**frozen)
    assert not (frozen["repository"] / "revision_work/phase5_gate.json").exists()
