"""Synthetic-only checks; no real protected region is opened or hashed."""

import copy
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from revision.guard import (
    HoldoutGuard,
    Phase5AlreadyConsumed,
    ProtocolViolation,
    begin_phase5,
    finish_phase5,
    positive_intersection,
    sha256_file,
)


@pytest.fixture
def protocol():
    return {
        "schema_version": 1,
        "regions": [
            {
                "id": "synthetic_holdout",
                # Source exists only as text. A guard must never open it.
                "source_candidate": "/DO_NOT_OPEN_SYNTHETIC_HOLDOUT.las",
                "protected_bounds": [100, 0, 110, 100],
                "scoring_bounds": [102, 10, 108, 20],
            }
        ],
        "development": {
            "allowed_context_bounds": [0, 0, 100, 100],
            "scoring_bounds": [10, 10, 20, 20],
        },
    }


def write_guard(tmp_path, protocol):
    holdout = tmp_path / "holdout.json"
    objective = tmp_path / "objective.md"
    holdout.write_text(json.dumps(protocol))
    objective.write_text("Minimize class-balanced loss; abstention and wrong cost 1.\n")
    return HoldoutGuard.from_files(
        holdout,
        objective,
        expected_holdout_sha256=sha256_file(holdout),
        expected_objective_sha256=sha256_file(objective),
    )


@pytest.fixture
def guard(tmp_path, protocol):
    return write_guard(tmp_path, protocol)


def test_development_bounds_allow_internal_crop_and_exact_boundary(guard):
    assert guard.assert_development_bounds([10, 10, 20, 20]) == (10, 10, 20, 20)
    assert guard.assert_development_bounds([99, 0, 100, 1]) == (99, 0, 100, 1)


@pytest.mark.parametrize(
    "bounds",
    [
        [99, 0, math.nextafter(100, math.inf), 1],
        [100, 0, 101, 1],
        [0, 0, 200, 200],
        [99, 99, 101, 101],
    ],
)
def test_any_positive_protected_overlap_fails(guard, bounds):
    with pytest.raises(ProtocolViolation, match="overlap protected"):
        guard.assert_development_bounds(bounds)


@pytest.mark.parametrize("bounds", [[-1, 0, 1, 1], [0, 99, 1, 101], [110, 0, 120, 1]])
def test_unprotected_but_unapproved_context_also_fails(guard, bounds):
    with pytest.raises(ProtocolViolation, match="outside permitted"):
        guard.assert_development_bounds(bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        [0, 0, 0, 1],
        [2, 0, 1, 1],
        [0, 2, 1, 1],
        [0, 0, float("nan"), 1],
        [0, 0, float("inf"), 1],
        [False, 0, 1, 1],
        ["0", 0, 1, 1],
        [0, 1, 2],
        "0001",
        None,
    ],
)
def test_invalid_bounds_fail_closed(guard, bounds):
    with pytest.raises(ProtocolViolation):
        guard.assert_development_bounds(bounds)


def test_half_open_edge_and_corner_do_not_overlap():
    assert not positive_intersection([0, 0, 1, 1], [1, 0, 2, 1])
    assert not positive_intersection([0, 0, 1, 1], [1, 1, 2, 2])
    assert positive_intersection([0, 0, 1, 1], [0, 0, 1, 1])


@pytest.mark.parametrize("name", ["holdout_path", "objective_path"])
def test_mutating_either_locked_file_blocks_development(guard, name):
    getattr(guard, name).write_bytes(getattr(guard, name).read_bytes() + b"\n")
    with pytest.raises(ProtocolViolation, match="changed"):
        guard.assert_development_bounds([10, 10, 20, 20])


def test_expected_hashes_cannot_be_omitted_or_invented(tmp_path, protocol):
    guard = write_guard(tmp_path, protocol)
    with pytest.raises(ProtocolViolation, match="SHA-256"):
        HoldoutGuard.from_files(
            guard.holdout_path,
            guard.objective_path,
            expected_holdout_sha256="",
            expected_objective_sha256="",
        )
    with pytest.raises(ProtocolViolation, match="changed"):
        HoldoutGuard.from_files(
            guard.holdout_path,
            guard.objective_path,
            expected_holdout_sha256="0" * 64,
            expected_objective_sha256=guard.expected_objective_sha256,
        )


@pytest.mark.parametrize(
    "change",
    ["empty", "duplicate", "scoring_escape", "context_overlap", "development_escape"],
)
def test_malformed_protocol_rejected(tmp_path, protocol, change):
    p = copy.deepcopy(protocol)
    if change == "empty":
        p["regions"] = []
    elif change == "duplicate":
        p["regions"] *= 2
    elif change == "scoring_escape":
        p["regions"][0]["scoring_bounds"][0] = 99
    elif change == "context_overlap":
        p["development"]["allowed_context_bounds"][2] = 101
    else:
        p["development"]["scoring_bounds"][0] = -1
    with pytest.raises(ProtocolViolation):
        write_guard(tmp_path, p)


def git(path, *args):
    return (
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )


@pytest.fixture
def frozen(tmp_path, guard):
    # A fresh synthetic Git repository; no changes to the user's Git history.
    git(tmp_path, "init", "-q")
    manifest = tmp_path / "final_manifest.json"
    manifest.write_text(json.dumps({"configuration": "synthetic", "source_hashes": {}}))
    git(tmp_path, "add", "holdout.json", "objective.md", "final_manifest.json")
    git(
        tmp_path,
        "-c",
        "user.name=Synthetic Test",
        "-c",
        "user.email=synthetic@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Freeze synthetic protocol",
    )
    return {
        "guard": guard,
        "manifest_path": manifest,
        "repository": tmp_path,
        "final_commit": git(tmp_path, "rev-parse", "HEAD"),
    }


def test_phase5_records_commit_hashes_and_annotation_only_mode(tmp_path, frozen):
    gate = tmp_path / "phase5.json"
    result = begin_phase5(gate, **frozen, mode="annotation_only")
    assert result["status"] == "started" and result["irreversible"]
    assert result["mode"] == "annotation_only"
    assert result["manifest_sha256"] == sha256_file(frozen["manifest_path"])
    assert result["holdout_sha256"] == frozen["guard"].expected_holdout_sha256
    assert result["objective_sha256"] == frozen["guard"].expected_objective_sha256
    assert result["final_commit"] == frozen["final_commit"]
    with pytest.raises(Phase5AlreadyConsumed):
        begin_phase5(gate, **frozen, mode="evaluation")


def test_phase5_crash_or_empty_reserved_file_is_not_retryable(tmp_path, frozen):
    gate = tmp_path / "phase5.json"
    gate.touch()
    with pytest.raises(Phase5AlreadyConsumed):
        begin_phase5(gate, **frozen)
    assert gate.read_bytes() == b""


def test_phase5_symlink_cannot_be_overwritten(tmp_path, frozen):
    gate = tmp_path / "phase5.json"
    gate.symlink_to(tmp_path / "nonexistent")
    with pytest.raises(Phase5AlreadyConsumed):
        begin_phase5(gate, **frozen)
    assert os.path.islink(gate)


def test_phase5_rejects_uncommitted_manifest_before_consuming_gate(tmp_path, frozen):
    frozen["manifest_path"].write_text('{"configuration": "changed"}')
    gate = tmp_path / "phase5.json"
    with pytest.raises(ProtocolViolation, match="differs from committed"):
        begin_phase5(gate, **frozen)
    assert not gate.exists()


def test_phase5_rejects_untracked_manifest_and_wrong_head(tmp_path, frozen):
    gate = tmp_path / "phase5.json"
    with pytest.raises(ProtocolViolation, match="HEAD"):
        begin_phase5(gate, **{**frozen, "final_commit": "0" * 40})
    untracked = tmp_path / "untracked.json"
    untracked.write_text('{"configuration": "uncommitted"}')
    with pytest.raises(ProtocolViolation, match="Git verification"):
        begin_phase5(gate, **{**frozen, "manifest_path": untracked})
    assert not gate.exists()


def test_phase5_cannot_replace_protocol_with_newly_matching_worktree_hash(
    tmp_path, frozen
):
    original = frozen["guard"]
    original.objective_path.write_text("Changed objective\n")
    altered = HoldoutGuard.from_files(
        original.holdout_path,
        original.objective_path,
        expected_holdout_sha256=original.expected_holdout_sha256,
        expected_objective_sha256=sha256_file(original.objective_path),
    )
    with pytest.raises(ProtocolViolation, match="not preserved in final commit"):
        begin_phase5(tmp_path / "phase5.json", **{**frozen, "guard": altered})


def test_phase5_atomic_exclusive_creation_allows_one_winner(tmp_path, frozen):
    gate = tmp_path / "phase5.json"

    def attempt(_):
        try:
            return begin_phase5(gate, **frozen)["event_id"]
        except Phase5AlreadyConsumed:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, range(8)))
    winners = [value for value in results if value is not None]
    assert len(winners) == 1
    assert json.loads(gate.read_text())["event_id"] == winners[0]


@pytest.mark.parametrize("status", ["completed", "blocked_no_reference", "failed"])
def test_terminal_status_is_append_only_and_does_not_reopen_gate(
    tmp_path, frozen, status
):
    gate = tmp_path / "phase5.json"
    event = begin_phase5(gate, **frozen)
    original = gate.read_bytes()
    finish = finish_phase5(
        gate, event_id=event["event_id"], status=status, details={"synthetic": True}
    )
    assert finish["status"] == status
    assert finish["gate_sha256"] == sha256_file(gate)
    assert gate.read_bytes() == original
    with pytest.raises(Phase5AlreadyConsumed):
        finish_phase5(gate, event_id=event["event_id"], status=status)
    with pytest.raises(Phase5AlreadyConsumed):
        begin_phase5(gate, **frozen)


def test_terminal_status_requires_matching_existing_reservation(tmp_path, frozen):
    gate = tmp_path / "phase5.json"
    with pytest.raises(ProtocolViolation, match="missing or unreadable"):
        finish_phase5(gate, event_id="unknown", status="failed")
    event = begin_phase5(gate, **frozen)
    with pytest.raises(ProtocolViolation, match="identity"):
        finish_phase5(gate, event_id="wrong", status="failed")
    with pytest.raises(ProtocolViolation, match="Invalid terminal"):
        finish_phase5(gate, event_id=event["event_id"], status="retry")
    assert not (tmp_path / "phase5.json.completion.json").exists()
