"""Fail-closed protocol locks and the irreversible Phase 5 access gate.

This module reads protocol/configuration files only. It never opens point clouds,
images, annotations, or prediction files. Callers must check development bounds
*before* opening a candidate input, and consume the final gate before any holdout
access. A failed/crashed Phase 5 still consumes the gate; there is no reset API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProtocolViolation(ValueError):
    """An immutable protocol or permitted spatial scope was violated."""


class Phase5AlreadyConsumed(ProtocolViolation):
    """The held-out access event has already started, even if it failed."""


def sha256_file(path: str | Path) -> str:
    """Hash a specified configuration file; never discover or scan data files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ProtocolViolation(f"{name} must be an explicit lowercase SHA-256")
    return value


def _bounds(value: Sequence[float], name: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ProtocolViolation(f"{name} must be [xmin, ymin, xmax, ymax]")
    try:
        raw = tuple(value)
        if len(raw) != 4 or any(isinstance(v, (bool, str, bytes)) for v in raw):
            raise ValueError
        bounds = tuple(float(v) for v in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolViolation(
            f"{name} must contain four numeric coordinates"
        ) from exc
    if not all(math.isfinite(v) for v in bounds):
        raise ProtocolViolation(f"{name} must contain finite coordinates")
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ProtocolViolation(f"{name} must have strictly positive width and height")
    return bounds


def positive_intersection(first: Sequence[float], second: Sequence[float]) -> bool:
    """Half-open rectangles overlap iff the intersection has positive area.

    No tolerance is used: even the smallest representable positive intrusion
    fails. Merely touching an edge or corner does not expose protected points.
    """
    a, b = _bounds(first, "first bounds"), _bounds(second, "second bounds")
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _contains(outer: Sequence[float], inner: Sequence[float]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


@dataclass(frozen=True)
class HoldoutGuard:
    holdout_path: Path
    objective_path: Path
    expected_holdout_sha256: str
    expected_objective_sha256: str

    @classmethod
    def from_files(
        cls,
        holdout_path: str | Path,
        objective_path: str | Path,
        *,
        expected_holdout_sha256: str,
        expected_objective_sha256: str,
    ) -> HoldoutGuard:
        """Load only with independently recorded preregistration hashes.

        Do not compute these expected values from current files on each run;
        store them in the manifest from the preregistered Git blobs.
        """
        guard = cls(
            Path(holdout_path).resolve(),
            Path(objective_path).resolve(),
            _digest(expected_holdout_sha256, "holdout hash"),
            _digest(expected_objective_sha256, "objective hash"),
        )
        guard.verify_locked_files()
        guard._protocol()
        return guard

    def verify_locked_files(self) -> dict[str, str]:
        expected = {
            "holdout": (self.holdout_path, self.expected_holdout_sha256),
            "objective": (self.objective_path, self.expected_objective_sha256),
        }
        result = {}
        for name, (path, digest) in expected.items():
            try:
                actual = sha256_file(path)
            except OSError as exc:
                raise ProtocolViolation(f"Cannot verify locked {name}: {path}") from exc
            if actual != digest:
                raise ProtocolViolation(
                    f"Locked {name} changed: expected {digest}, got {actual}"
                )
            result[name + "_sha256"] = actual
        return result

    def _protocol(self) -> dict[str, Any]:
        # Hash the same bytes parsed, eliminating a check-then-read race here.
        try:
            raw = self.holdout_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.expected_holdout_sha256:
                raise ProtocolViolation("Locked holdout changed while reading")
            protocol = json.loads(raw)
            if (
                type(protocol["schema_version"]) is not int
                or protocol["schema_version"] != 1
            ):
                raise ProtocolViolation("Unsupported holdout protocol schema")
            regions = protocol["regions"]
            if not isinstance(regions, list) or not regions:
                raise ProtocolViolation("At least one protected region is required")
            identifiers = set()
            for region in regions:
                identifier = region["id"]
                if (
                    not isinstance(identifier, str)
                    or not identifier
                    or identifier in identifiers
                ):
                    raise ProtocolViolation(
                        "Protected region IDs must be unique nonempty strings"
                    )
                identifiers.add(identifier)
                protected = _bounds(
                    region["protected_bounds"], f"{identifier} protected bounds"
                )
                scoring = _bounds(
                    region["scoring_bounds"], f"{identifier} scoring bounds"
                )
                if not _contains(protected, scoring):
                    raise ProtocolViolation(
                        f"{identifier} scoring bounds extend beyond protection"
                    )
            context = _bounds(
                protocol["development"]["allowed_context_bounds"], "development context"
            )
            scoring = _bounds(
                protocol["development"]["scoring_bounds"], "development scoring bounds"
            )
            if not _contains(context, scoring):
                raise ProtocolViolation(
                    "Development scoring bounds extend beyond allowed context"
                )
            for region in regions:
                if positive_intersection(context, region["protected_bounds"]):
                    raise ProtocolViolation(
                        "Development context overlaps protected holdout"
                    )
            return protocol
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProtocolViolation("Missing or malformed holdout protocol") from exc

    def assert_development_bounds(
        self, bounds: Sequence[float]
    ) -> tuple[float, float, float, float]:
        """Reject protected overlap and any scope outside preregistered context."""
        self.verify_locked_files()
        requested = _bounds(bounds, "requested development bounds")
        protocol = self._protocol()
        for region in protocol["regions"]:
            if positive_intersection(requested, region["protected_bounds"]):
                raise ProtocolViolation(
                    f"Development bounds overlap protected region {region['id']}"
                )
        if not _contains(protocol["development"]["allowed_context_bounds"], requested):
            raise ProtocolViolation(
                "Development bounds extend outside permitted development context"
            )
        return requested


def _git(repository: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolViolation(f"Git verification failed: {' '.join(args)}") from exc


def _committed_hash(repository: Path, path: Path, commit: str) -> str:
    try:
        relative = path.resolve().relative_to(repository).as_posix()
    except ValueError as exc:
        raise ProtocolViolation(f"Frozen file is outside repository: {path}") from exc
    committed = _git(repository, "show", f"{commit}:{relative}")
    return hashlib.sha256(committed).hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably reserve an event; never truncate or replace an existing path."""
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise Phase5AlreadyConsumed(
            f"Irreversible Phase 5 record already exists: {path}"
        ) from exc
    # A write error deliberately leaves the reserved file in place. Recovery
    # requires a new human-approved protocol, never deletion and a second run.
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def begin_phase5(
    gate_path: str | Path,
    *,
    guard: HoldoutGuard,
    manifest_path: str | Path,
    repository: str | Path,
    final_commit: str,
    mode: str = "evaluation",
) -> dict[str, Any]:
    """Consume Phase 5 once after proving the final manifest is committed.

    This must succeed before inspecting any holdout data. `annotation_only` is
    also a consumed event, not permission to later score under the same gate.
    Unrelated worktree changes do not invalidate a committed final manifest.
    """
    gate_path = Path(gate_path)
    if os.path.lexists(gate_path) or os.path.lexists(
        str(gate_path) + ".completion.json"
    ):
        raise Phase5AlreadyConsumed(f"Phase 5 has already been consumed: {gate_path}")
    if mode not in {"evaluation", "annotation_only"}:
        raise ProtocolViolation("Phase 5 mode must be evaluation or annotation_only")
    if not isinstance(final_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", final_commit
    ):
        raise ProtocolViolation("final_commit must be a full Git commit hash")
    repository = Path(repository).resolve()
    resolved_root = Path(
        _git(repository, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if resolved_root != repository:
        raise ProtocolViolation("repository must name the Git worktree root")
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    if head != final_commit:
        raise ProtocolViolation("HEAD does not equal the frozen final commit")
    locks = guard.verify_locked_files()
    guard._protocol()
    for path, expected in [
        (guard.holdout_path, guard.expected_holdout_sha256),
        (guard.objective_path, guard.expected_objective_sha256),
    ]:
        if _committed_hash(repository, path, final_commit) != expected:
            raise ProtocolViolation(
                f"Locked protocol is not preserved in final commit: {path}"
            )
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolViolation(
            "Final manifest is missing or is not valid JSON"
        ) from exc
    if not isinstance(manifest, dict) or not manifest:
        raise ProtocolViolation("Final manifest must be a nonempty JSON object")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if _committed_hash(repository, manifest_path, final_commit) != manifest_hash:
        raise ProtocolViolation("Final manifest differs from committed bytes")
    event = {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "mode": mode,
        "final_commit": final_commit,
        "manifest_path": manifest_path.relative_to(repository).as_posix(),
        "manifest_sha256": manifest_hash,
        **locks,
        "irreversible": True,
        "completion_record": gate_path.name + ".completion.json",
        "recovery_rule": "Failure or missing reference labels still consumes this event; no retry under this protocol.",
    }
    _exclusive_json(gate_path, event)
    return event


def finish_phase5(
    gate_path: str | Path,
    *,
    event_id: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one completion record without changing the original reservation."""
    if status not in {"completed", "blocked_no_reference", "failed"}:
        raise ProtocolViolation("Invalid terminal Phase 5 status")
    gate_path = Path(gate_path)
    try:
        gate_bytes = gate_path.read_bytes()
        event = json.loads(gate_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolViolation(
            "Cannot complete a missing or unreadable Phase 5 reservation"
        ) from exc
    if (
        event.get("event_id") != event_id
        or event.get("status") != "started"
        or event.get("irreversible") is not True
    ):
        raise ProtocolViolation("Phase 5 reservation identity does not match")
    completion = {
        "schema_version": 1,
        "event_id": event_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "details": details or {},
    }
    _exclusive_json(Path(str(gate_path) + ".completion.json"), completion)
    return completion
