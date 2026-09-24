"""One-shot Phase 5 annotation branch; no inference or scoring is implemented.

The final manifest and its gate/output paths must be committed. Only after the
irreversible gate starts may this command inspect the locked candidate reference
files and create prediction-hidden annotation inputs. An unapproved nonzero LAS
classification is never accepted as independent human ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import laspy
import numpy as np

from revision.annotation_package import export_holdout_packages
from revision.guard import (
    HoldoutGuard,
    ProtocolViolation,
    _committed_hash,
    _exclusive_json,
    _git,
    begin_phase5,
    finish_phase5,
    sha256_file,
)

UNKNOWN = "UNKNOWN - needs provider/PI"
HUMAN_REQUIREMENT = (
    "Provide independent prediction-blind human reference annotations for the "
    "preregistered held-out rectangles, with immutable point-ID mapping, source "
    "hash, taxonomy, coverage/uncertainty rules, annotator identities, dates, "
    "adjudication history and explicit PI/researcher approval. Existing nonzero "
    "LAS classes alone are insufficient. A later evaluation requires a new "
    "explicitly approved protocol; this consumed Phase 5 gate cannot be rerun."
)


def _path(path):
    return Path(os.path.abspath(path))


def _bound_path(repository, value, name):
    if not isinstance(value, str):
        raise ProtocolViolation(f"Final manifest must bind {name} as a relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ProtocolViolation(f"Invalid frozen {name} path")
    return repository / relative


def _inspect_candidates(repository, protocol):
    """Called only after begin_phase5 succeeds; read exact locked candidates."""
    records = []
    for region in protocol["regions"]:
        candidate = Path(region["source_candidate"])
        if candidate.parent != Path("data/eval") or not candidate.name.endswith(
            "_gt.las"
        ):
            raise ProtocolViolation(
                "Locked reference candidate is outside the authorized candidate naming convention"
            )
        path = repository / candidate
        record = {
            "region_id": region["id"],
            "candidate": candidate.as_posix(),
            "scoring_bounds": region["scoring_bounds"],
            "independent_human_approval": "NOT PROVIDED in committed final manifest",
            "annotation_provenance": UNKNOWN,
            "eligible_for_scoring": False,
        }
        if not path.exists():
            record.update({"exists": False, "status": "candidate file absent"})
            records.append(record)
            continue
        if path.is_symlink():
            raise ProtocolViolation("Locked candidate reference must not be a symlink")
        before = sha256_file(path)
        bounds = region["scoring_bounds"]
        counts = np.zeros(256, dtype=np.int64)
        with laspy.open(path) as reader:
            total = reader.header.point_count
            for points in reader.chunk_iterator(200_000):
                x, y = np.asarray(points.x), np.asarray(points.y)
                inside = (
                    (x >= bounds[0])
                    & (x < bounds[2])
                    & (y >= bounds[1])
                    & (y < bounds[3])
                )
                classifications = np.asarray(points.classification)[inside]
                counts += np.bincount(classifications, minlength=256)
        if sha256_file(path) != before:
            raise ProtocolViolation(
                "Candidate reference changed during read-only inventory"
            )
        non_placeholder = int(counts.sum() - counts[0] - counts[1])
        record.update(
            {
                "exists": True,
                "source_sha256_before": before,
                "source_sha256_after": before,
                "candidate_file_points": int(total),
                "candidate_scoring_rectangle_points": int(counts.sum()),
                "candidate_classification_counts": {
                    str(i): int(n) for i, n in enumerate(counts) if n
                },
                "non_placeholder_classification_count": non_placeholder,
                "status": (
                    "nonzero candidate classes present, but independent provenance/approval UNKNOWN; not reference evidence"
                    if non_placeholder
                    else "only unclassified/unassigned candidate points; no independent references established"
                ),
                "counts_interpretation": "stored candidate codes only, not reviewed class counts or accuracy",
                "reference_unchanged": True,
            }
        )
        records.append(record)
    return records


def run_final_gate(
    *, repository, manifest_path, final_commit, gate_path=None, output_dir=None
):
    """Consume once, inventory unapproved references, export blind input, stop."""
    repository = _path(repository)
    manifest_path = _path(manifest_path)
    if manifest_path.name != "final_manifest.json":
        raise ProtocolViolation(
            "Phase 5 requires the explicitly named final_manifest.json"
        )
    manifest = json.loads(manifest_path.read_text())
    phase5 = manifest.get("phase5")
    if not isinstance(phase5, dict):
        raise ProtocolViolation(
            "Final manifest lacks the committed phase5 path/reference policy"
        )
    approved = phase5.get("approved_reference_records")
    if not isinstance(approved, list):
        raise ProtocolViolation(
            "phase5.approved_reference_records must explicitly be a list"
        )
    if approved:
        # This narrow CLI cannot validate a new human approval record or silently
        # take an inference branch. Do not consume a gate before that is resolved.
        raise ProtocolViolation(
            "This CLI implements only the no-approved-reference annotation branch. "
            "Nonempty approval records require verification and a frozen evaluation "
            "implementation before any Phase 5 access. " + HUMAN_REQUIREMENT
        )
    frozen_gate = _bound_path(repository, phase5.get("gate_path"), "gate_path")
    frozen_output = _bound_path(
        repository, phase5.get("annotation_output"), "annotation_output"
    )
    if gate_path is not None and _path(gate_path) != frozen_gate:
        raise ProtocolViolation("Gate path differs from the committed final manifest")
    if output_dir is not None and _path(output_dir) != frozen_output:
        raise ProtocolViolation(
            "Annotation output differs from the committed final manifest"
        )
    if frozen_output.exists():
        raise FileExistsError(
            f"Refusing to overwrite annotation output: {frozen_output}"
        )
    # Validate the code that will perform holdout access, without touching a
    # data path. Unrelated preserved worktree changes remain permissible.
    if _git(repository, "rev-parse", "HEAD").decode().strip() != final_commit:
        raise ProtocolViolation("HEAD does not equal the requested final commit")
    for name in (
        "revision/final_gate.py",
        "revision/annotation_package.py",
        "revision/guard.py",
    ):
        expected = manifest.get("source_hashes", {}).get(name)
        if not expected or sha256_file(repository / name) != expected:
            raise ProtocolViolation(
                f"Phase 5 executable source differs from final manifest: {name}"
            )
        if _committed_hash(repository, repository / name, final_commit) != expected:
            raise ProtocolViolation(
                f"Phase 5 executable source is not frozen in final commit: {name}"
            )
    guard = HoldoutGuard.from_files(
        repository / "holdout.json",
        repository / "objective.md",
        expected_holdout_sha256=manifest["protocol"]["holdout_sha256"],
        expected_objective_sha256=manifest["protocol"]["objective_sha256"],
    )
    protocol = guard._protocol()
    # The only begin_phase5 invocation in this command. Everything above reads
    # configuration/output metadata only; no candidate/raw held-out paths touched.
    event = begin_phase5(
        frozen_gate,
        guard=guard,
        manifest_path=manifest_path,
        repository=repository,
        final_commit=final_commit,
        mode="annotation_only",
    )
    inventory_path = Path(str(frozen_gate) + ".reference_inventory.json")
    try:
        inventory = _inspect_candidates(repository, protocol)
        _exclusive_json(
            inventory_path,
            {
                "schema_version": 1,
                "event_id": event["event_id"],
                "records": inventory,
                "independent_reference_status": "NOT ESTABLISHED",
                "required_human_action": HUMAN_REQUIREMENT,
                "model_inference_run": False,
                "heldout_accuracy": None,
            },
        )
        package = export_holdout_packages(
            repository=repository,
            gate_path=frozen_gate,
            manifest_path=manifest_path,
            output_dir=frozen_output,
            guard=guard,
            event_id=event["event_id"],
        )
        details = {
            "reason": "No independently approved held-out reference package supplied",
            "reference_inventory": str(inventory_path.relative_to(repository)),
            "reference_inventory_sha256": sha256_file(inventory_path),
            "annotation_package": package,
            "required_human_action": HUMAN_REQUIREMENT,
            "candidate_references_modified": False,
            "model_inference_run": False,
            "heldout_accuracy": None,
        }
    except Exception as exc:
        finish_phase5(
            frozen_gate,
            event_id=event["event_id"],
            status="failed",
            details={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_inference_run": False,
                "heldout_accuracy": None,
                "required_human_action": "Inspect the preserved failure artifacts; never retry or delete this consumed gate. "
                + HUMAN_REQUIREMENT,
            },
        )
        raise
    completion = finish_phase5(
        frozen_gate,
        event_id=event["event_id"],
        status="blocked_no_reference",
        details=details,
    )
    return {
        "event_id": event["event_id"],
        "status": completion["status"],
        "gate_path": str(frozen_gate),
        "annotation_output": str(frozen_output),
        "reference_inventory": str(inventory_path),
        "heldout_accuracy": None,
        "model_inference_run": False,
        "required_human_action": HUMAN_REQUIREMENT,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--final-commit", required=True)
    parser.add_argument("--gate-path")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run_final_gate(
        repository=args.repository,
        manifest_path=args.manifest,
        final_commit=args.final_commit,
        gate_path=args.gate_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
