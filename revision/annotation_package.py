"""Create new blind annotation inputs, never semantic reference labels.

Pilot packages use only the completed development run and raw tile C. Held-out
sources cannot be opened until a matching one-shot Phase 5 reservation exists;
an additional exclusive export claim prevents repeating annotation preparation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import laspy
import numpy as np

from revision.guard import (
    HoldoutGuard,
    ProtocolViolation,
    _committed_hash,
    _exclusive_json,
    _git,
    sha256_file,
)

UNKNOWN = "UNKNOWN - needs provider/PI"
MAX_POINTS = 2**24
SEED = 20260923
OBSERVED_FIELDS = ("intensity", "return_number", "number_of_returns", "red", "green", "blue")


def _json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _path(path):
    # Unlike resolve(), normalization does not inspect the candidate data path.
    return Path(os.path.abspath(path))


def _inside(x, y, bounds):
    return (x >= bounds[0]) & (y >= bounds[1]) & (x < bounds[2]) & (y < bounds[3])


def _ids(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ProtocolViolation(f"{name} must be nonnegative integer IDs")
    if len(np.unique(values)) != len(values):
        raise ProtocolViolation(f"{name} contains duplicate IDs")
    return values.astype(np.uint64)


def _field_digest(points, fields, dtype):
    return np.column_stack([np.asarray(points[field]) for field in fields]).astype(dtype).tobytes()


def _audit_sample(arrays, context_bounds):
    """Select input-only strata; do not use prediction or class for strata."""
    ids = _ids(arrays["ids"], "pilot IDs")
    n = len(ids)
    x, y, z, hag = [np.asarray(arrays[k]) for k in ("x", "y", "z", "hag")]
    rgb = np.asarray(arrays["rgb"])
    ref = np.asarray(arrays["reference"])
    if any(a.shape != (n,) for a in (x, y, z, hag, ref)) or rgb.shape != (n, 3):
        raise ProtocolViolation("Pilot audit arrays have incompatible shapes")
    if not np.isfinite(np.column_stack([x, y, z, rgb])).all():
        raise ProtocolViolation("Pilot XYZ/RGB must be finite")
    if not _inside(x, y, context_bounds).all():
        raise ProtocolViolation("Pilot points lie outside permitted context")
    if not np.isin(ref, [0, 1, 3, 5, 6, 11, 64]).all():
        raise ProtocolViolation("Unsupported reference membership code")
    reviewed = np.isin(ref, [3, 5, 6, 11, 64])
    if not reviewed.any():
        raise ProtocolViolation("There are no reviewed pilot IDs to audit")
    rgbf = rgb.astype(np.float64)
    total = rgbf.sum(axis=1)
    color_valid = (total > 0) & (rgbf >= 0).all(axis=1) & (rgbf <= 65535).all(axis=1)
    exg = np.full(n, np.nan)
    np.divide(2 * rgbf[:, 1] - rgbf[:, 0] - rgbf[:, 2], total, out=exg, where=color_valid)
    resolution = 0.5
    cols = int(np.ceil((context_bounds[2] - context_bounds[0]) / resolution))
    rows = int(np.ceil((context_bounds[3] - context_bounds[1]) / resolution))
    if rows * cols > 2_000_000:
        raise ProtocolViolation("Audit context grid exceeds bounded allocation")
    c = np.floor((x - context_bounds[0]) / resolution).astype(int)
    r = np.floor((y - context_bounds[1]) / resolution).astype(int)
    flat = r * cols + c
    count = np.bincount(flat, minlength=rows * cols).reshape(rows, cols)
    height = np.full(rows * cols, -np.inf)
    np.maximum.at(height, flat, z)
    height = height.reshape(rows, cols)
    ec = np.bincount(flat[color_valid], minlength=rows * cols)
    es = np.bincount(flat[color_valid], weights=exg[color_valid], minlength=rows * cols)
    color = np.full(rows * cols, np.nan)
    np.divide(es, ec, out=color, where=ec > 0)
    color = color.reshape(rows, cols)
    occupied = count > 0
    padded_count = np.pad(occupied, 1, constant_values=False)
    padded_z = np.pad(height, 1, constant_values=np.nan)
    padded_e = np.pad(color, 1, constant_values=np.nan)
    empty_neighbor = np.zeros_like(occupied)
    zmin = np.full_like(height, np.inf)
    zmax = np.full_like(height, -np.inf)
    emin = np.full_like(height, np.inf)
    emax = np.full_like(height, -np.inf)
    for dr in range(3):
        for dc in range(3):
            occ = padded_count[dr:dr + rows, dc:dc + cols]
            zv = padded_z[dr:dr + rows, dc:dc + cols]
            ev = padded_e[dr:dr + rows, dc:dc + cols]
            if (dr, dc) != (1, 1):
                empty_neighbor |= ~occ
            zmin = np.minimum(zmin, np.where(np.isfinite(zv), zv, np.inf))
            zmax = np.maximum(zmax, np.where(np.isfinite(zv), zv, -np.inf))
            emin = np.minimum(emin, np.where(np.isfinite(ev), ev, np.inf))
            emax = np.maximum(emax, np.where(np.isfinite(ev), ev, -np.inf))
    height_boundary = zmax - zmin > 2.0
    color_boundary = emax - emin > 0.05
    boundary = (empty_neighbor | height_boundary | color_boundary)[r, c]
    hbin = np.where(~np.isfinite(hag), 3, np.where(hag < 2, 0, np.where(hag <= 6, 1, 2)))
    ebin = np.where(~color_valid, 2, np.where(exg <= 0.05, 0, 1))
    strata = (hbin * 3 + ebin) * 2 + boundary.astype(int)
    rng = np.random.default_rng(SEED)
    selected = []
    records = []
    for stratum in range(24):
        candidates = np.flatnonzero(reviewed & (strata == stratum))
        candidates = candidates[np.argsort(ids[candidates], kind="stable")]
        nh = min(50, len(candidates))
        chosen = rng.choice(candidates, size=nh, replace=False) if nh else np.empty(0, dtype=int)
        selected.extend(chosen.tolist())
        records.append({"stratum": stratum, "N_h": len(candidates), "n_h": nh,
                        "inclusion_probability": nh / len(candidates) if len(candidates) else None})
    selected = np.asarray(selected, dtype=int)
    return ids[selected], {
        "seed": SEED, "target_count": len(selected), "max_targets": 1200,
        "strata": records, "resolution_ft": resolution, "grid_origin": list(context_bounds[:2]),
        "context_available": "pilot_arrays geometry only; absent neighboring context is empty for boundary sampling",
        "boundary_trigger_point_counts": {
            "empty_neighbor": int(empty_neighbor[r, c].sum()),
            "height_range_gt_2ft": int(height_boundary[r, c].sum()),
            "exg_range_gt_0_05": int(color_boundary[r, c].sum()),
        },
        "strata_source": "HAG, original RGB, geometric/color boundaries; reference used only for reviewed membership",
    }


def _blank_csv(path, indices):
    with Path(path).open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["local_id", "annotation_status", "semantic_class", "reason", "annotator_id", "annotation_time"])
        for local_id in indices:
            writer.writerow([int(local_id), "", "", "", "", ""])


def _instructions(path, *, audit):
    Path(path).write_text(
        "# Prediction-hidden annotation input\n\n"
        "This is an UNLABELLED ANNOTATION TASK, not ground truth and not a prediction. "
        "Every exported classification is 0 (never classified); user data and semantic flags are cleared. "
        "Classify from observed XYZ/RGB context, never from that placeholder.\n\n"
        "Use tile_index as local_id in the blank response CSV. Preserve integer IDs exactly. "
        "Return your response as a separate file; never overwrite this input or any source file. "
        "The private identity key, original labels, model outputs and sampling strata are withheld.\n\n"
        + ("audit_target=1 identifies a sampled target; other points provide unlabelled context. " if audit else "Every point is an annotation target. ")
        + "Leave semantic_class blank when annotation_status is uncertain or other. "
        "Record why a point is ambiguous; do not infer classes solely from height or greenness.\n\n"
        "Allowed confident semantic responses: 3 grass, 5 tree, 6 building, 11 hard surface, 64 vehicle. "
        "Low tree branches/trunks remain tree when identifiable; shrubs, soil, litter or equipment must not "
        "be forced into an incompatible class. Use other with a description when outside the taxonomy. "
        "Use uncertain when color registration, occlusion or boundaries prevent a reliable judgment.\n\n"
        "Two humans should annotate independently while blinded to each other's responses. "
        "A human adjudicator preserves both responses and records disagreement decisions. "
        "Software must not supply semantic labels or change the original reference.\n\n"
        "Coordinates retain source scales/offsets. Units follow the preregistered US survey foot assumption; "
        "provider/sensor/date/colorization provenance may be UNKNOWN - needs provider/PI.\n"
    )


def _export_region(source, root, region_id, bounds, *, target_ids=None, expected_arrays=None):
    """Stream a single authorized raw tile into a sanitized new annotation copy."""
    source = _path(source)
    if source.is_symlink():
        raise ProtocolViolation("Raw annotation source must not be a symlink")
    source_hash = sha256_file(source)
    public = root / "annotator" / region_id
    private = root / "private" / region_id
    public.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    output = public / "annotation_input.las"
    expected_digests = {k: hashlib.sha256() for k in ["xyz", "rgb", "observations"]}
    source_ids, source_rows = [], []
    target_local = []
    n = 0
    source_row = 0
    target_ids = None if target_ids is None else _ids(target_ids, "audit target IDs")
    with laspy.open(source) as reader:
        original_header = reader.header
        dimensions = set(original_header.point_format.dimension_names)
        if not {"red", "green", "blue"}.issubset(dimensions):
            raise ProtocolViolation("Original RGB is unavailable for blind annotation")
        has_original_id = "orig_index" in dimensions
        header = laspy.LasHeader(point_format=7, version="1.4")
        header.scales = original_header.scales.copy()
        header.offsets = original_header.offsets.copy()
        header.system_identifier = "ANNOTATION INPUT"
        header.generating_software = "blind annotation export"
        header.add_extra_dim(laspy.ExtraBytesParams(name="tile_index", type=np.uint32))
        if target_ids is not None:
            header.add_extra_dim(laspy.ExtraBytesParams(name="audit_target", type=np.uint8))
        try:
            crs = original_header.parse_crs()
        except (ImportError, ValueError):
            crs = None
        if crs is not None:
            header.add_crs(crs)
        with output.open("xb") as stream, laspy.open(stream, mode="w", header=header, closefd=False) as writer:
            for chunk in reader.chunk_iterator(200_000):
                selection = _inside(np.asarray(chunk.x), np.asarray(chunk.y), bounds)
                count = int(selection.sum())
                if count:
                    if n + count > MAX_POINTS:
                        raise ProtocolViolation("Annotation package exceeds exact float32 local-ID limit; stop without retry")
                    chosen = chunk[selection]
                    rows = np.flatnonzero(selection).astype(np.uint64) + source_row
                    ids = np.asarray(chosen.orig_index) if has_original_id else rows
                    if ids.dtype.kind not in "iu" or np.any(ids < 0):
                        raise ProtocolViolation("Raw source identity must contain nonnegative integers")
                    ids = ids.astype(np.uint64)
                    clean = laspy.ScaleAwarePointRecord.zeros(count, header=header)
                    for field in ("X", "Y", "Z", *OBSERVED_FIELDS):
                        clean[field] = chosen[field]
                    local = np.arange(n, n + count, dtype=np.uint32)
                    clean.tile_index = local
                    # New records are zero-initialized: classification, user_data,
                    # semantic flags, source IDs and all non-allowlisted fields stay 0.
                    if target_ids is not None:
                        mask = np.isin(ids, target_ids)
                        clean.audit_target = mask.astype(np.uint8)
                        target_local.extend(local[mask].tolist())
                    expected_digests["xyz"].update(_field_digest(clean, ("X", "Y", "Z"), "<i4"))
                    expected_digests["rgb"].update(_field_digest(clean, ("red", "green", "blue"), "<u2"))
                    expected_digests["observations"].update(_field_digest(clean, OBSERVED_FIELDS[:3], "<u2"))
                    writer.write_points(clean)
                    source_ids.append(ids.copy())
                    source_rows.append(rows)
                    n += count
                source_row += len(chunk)
    if not n:
        raise ProtocolViolation("No points in preregistered annotation rectangle")
    originals = _ids(np.concatenate(source_ids), "exported original IDs")
    rows = np.concatenate(source_rows)
    if target_ids is not None and len(target_local) != len(target_ids):
        raise ProtocolViolation("Audit target IDs are missing from the exported pilot")
    if expected_arrays is not None:
        expected_ids = _ids(expected_arrays["ids"], "pilot array IDs")
        if len(expected_ids) != n or not np.array_equal(np.sort(originals), np.sort(expected_ids)):
            raise ProtocolViolation("Pilot arrays do not match the raw crop ID set")
    hashes = {k: v.hexdigest() for k, v in expected_digests.items()}
    actual_digests = {k: hashlib.sha256() for k in hashes}
    seen = 0
    with laspy.open(output) as reader:
        for chunk in reader.chunk_iterator(200_000):
            local = np.asarray(chunk.tile_index)
            if not np.array_equal(local, np.arange(seen, seen + len(chunk))):
                raise ProtocolViolation("Local-ID round trip failed")
            if np.any(chunk.classification) or np.any(chunk.user_data) or np.any(chunk.classification_flags):
                raise ProtocolViolation("Semantic attributes leaked into annotation input")
            if expected_arrays is not None:
                order = np.argsort(expected_arrays["ids"])
                pos = order[np.searchsorted(np.asarray(expected_arrays["ids"])[order], originals[seen:seen + len(chunk)])]
                for dimension in ("x", "y", "z"):
                    if not np.array_equal(np.asarray(getattr(chunk, dimension)), expected_arrays[dimension][pos]):
                        raise ProtocolViolation("Pilot array/raw coordinate correspondence failed")
                rgb = np.column_stack([chunk.red, chunk.green, chunk.blue])
                if not np.array_equal(rgb, expected_arrays["rgb"][pos]):
                    raise ProtocolViolation("Pilot array/raw RGB correspondence failed")
            actual_digests["xyz"].update(_field_digest(chunk, ("X", "Y", "Z"), "<i4"))
            actual_digests["rgb"].update(_field_digest(chunk, ("red", "green", "blue"), "<u2"))
            actual_digests["observations"].update(_field_digest(chunk, OBSERVED_FIELDS[:3], "<u2"))
            seen += len(chunk)
    if seen != n or hashes != {k: v.hexdigest() for k, v in actual_digests.items()}:
        raise ProtocolViolation("Annotation field round trip failed")
    if sha256_file(source) != source_hash:
        raise ProtocolViolation("Raw source changed during annotation export")
    with (private / "identity_map.npz").open("xb") as stream:
        np.savez_compressed(stream, local_id=np.arange(n, dtype=np.uint32), original_id=originals,
                            source_row=rows, source_sha256=np.asarray(source_hash))
    _blank_csv(public / "response_template.csv", range(n) if target_ids is None else sorted(target_local))
    _instructions(public / "INSTRUCTIONS.md", audit=target_ids is not None)
    record = {
        "schema_version": 1, "role": "new unlabelled annotation input; not reference labels",
        "region_id": region_id, "bounds": list(bounds), "bounds_convention": "half-open XY; all heights",
        "source_path": str(source), "source_sha256": source_hash,
        "source_identity": "orig_index" if has_original_id else "source-file row ordinal; no global ID available",
        "point_count": n, "target_count": n if target_ids is None else len(target_local),
        "classification_placeholder": 0, "identity_field": "tile_index",
        "source_reference_labels_read": False, "predictions_exported": False,
        "retained_fields": ["XYZ", "RGB", "intensity", "return_number", "number_of_returns"],
        "other_fields": "cleared or omitted; original full records remain in source",
        "coordinate_units": "US survey foot - preregistered assumption, not provider verification",
        "crs": str(crs) if crs is not None else UNKNOWN,
        "sensor": UNKNOWN, "acquisition_date": UNKNOWN, "colorization": UNKNOWN,
        "scale": header.scales.tolist(), "offset": header.offsets.tolist(),
        "field_hashes": hashes, "export_sha256": sha256_file(output),
        "identity_map_sha256": sha256_file(private / "identity_map.npz"),
        "round_trip": "unique integer IDs; exact raw XYZ, RGB and retained observations",
        "code_sha256": sha256_file(__file__),
    }
    _json(private / "manifest.json", record)
    # No source identity, hashes, strata, or join keys in the public record.
    _json(public / "package.json", {k: record[k] for k in [
        "schema_version", "role", "region_id", "bounds", "bounds_convention", "point_count",
        "target_count", "classification_placeholder", "identity_field", "coordinate_units", "crs",
        "sensor", "acquisition_date", "colorization", "scale", "offset",
    ]})
    return record


def export_pilot_audit(*, arrays_path, raw_tile_path, output_dir, guard):
    """Create a blinded development-label audit; never rewrite a reference."""
    guard.verify_locked_files()
    protocol = guard._protocol()
    bounds = protocol["development"]["scoring_bounds"]
    guard.assert_development_bounds(bounds)
    repository = guard.holdout_path.parent
    if _path(raw_tile_path) != repository / "data/eval/tile_c.las":
        raise ProtocolViolation("Pilot audit accepts only the fixed raw tile C input")
    arrays_path = _path(arrays_path)
    if arrays_path.name != "pilot_arrays.npz" or arrays_path.parent.name != "output":
        raise ProtocolViolation("Expected a completed run output/pilot_arrays.npz")
    manifest_path = arrays_path.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"]["input"] != "data/eval/tile_c.las":
        raise ProtocolViolation("Pilot manifest does not identify raw tile C")
    if manifest["protocol"] != {"holdout_sha256": guard.expected_holdout_sha256,
                                "objective_sha256": guard.expected_objective_sha256}:
        raise ProtocolViolation("Pilot manifest protocol hashes do not match")
    expected_hash = json.loads((arrays_path.parent / "artifact_hashes.json").read_text())["output/pilot_arrays.npz"]
    if sha256_file(arrays_path) != expected_hash:
        raise ProtocolViolation("Pilot arrays differ from completed-run artifact hash")
    if sha256_file(raw_tile_path) != manifest["dataset"]["input_sha256"]:
        raise ProtocolViolation("Raw tile C differs from completed-run source hash")
    # np.load is lazy: prediction/baseline/boundary fields are never accessed.
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in ["ids", "x", "y", "z", "rgb", "hag", "reference"]}
    if not _inside(arrays["x"], arrays["y"], bounds).all():
        raise ProtocolViolation("Pilot arrays extend outside preregistered scoring bounds")
    targets, sampling = _audit_sample(arrays, protocol["development"]["allowed_context_bounds"])
    root = _path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "private").mkdir(mode=0o700)
    record = _export_region(raw_tile_path, root, "pilot_c_audit", bounds,
                            target_ids=targets, expected_arrays=arrays)
    sampling["arrays_sha256"] = expected_hash
    sampling["source_manifest_sha256"] = sha256_file(manifest_path)
    _json(root / "private/pilot_c_audit/sampling.json", sampling)
    _json(root / "private/package_summary.json", {"kind": "pilot-audit", "regions": [record]})
    return {"kind": "pilot-audit", "output_dir": str(root), "point_count": record["point_count"],
            "target_count": len(targets), "source_unchanged": True, "labels_created": False}


def export_holdout_packages(*, repository, gate_path, manifest_path, output_dir, guard, event_id):
    """Use an already-consumed Phase 5 gate; do not open any GT/prediction file."""
    guard.verify_locked_files()
    protocol = guard._protocol()
    repository, gate_path, manifest_path = map(_path, [repository, gate_path, manifest_path])
    if repository != guard.holdout_path.parent:
        raise ProtocolViolation("Repository does not match locked protocol location")
    event = json.loads(gate_path.read_text())
    if (event.get("event_id") != event_id or event.get("status") != "started"
            or event.get("irreversible") is not True or event.get("mode") not in {"annotation_only", "evaluation"}):
        raise ProtocolViolation("Existing one-shot Phase 5 reservation does not match")
    if Path(str(gate_path) + ".completion.json").exists():
        raise ProtocolViolation("Phase 5 already completed; annotation export cannot reopen it")
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    if event["final_commit"] != head:
        raise ProtocolViolation("Phase 5 frozen commit no longer matches HEAD")
    if manifest_path != repository / event["manifest_path"]:
        raise ProtocolViolation("Phase 5 manifest path does not match reservation")
    if sha256_file(manifest_path) != event["manifest_sha256"] or _committed_hash(repository, manifest_path, head) != event["manifest_sha256"]:
        raise ProtocolViolation("Phase 5 manifest differs from frozen bytes")
    if event.get("holdout_sha256") != guard.expected_holdout_sha256 or event.get("objective_sha256") != guard.expected_objective_sha256:
        raise ProtocolViolation("Phase 5 protocol hashes differ")
    root = _path(output_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite annotation package: {root}")
    # Reserve the entire export once, before touching even metadata of raw A/B.
    _exclusive_json(Path(str(gate_path) + ".annotation_export.json"), {
        "schema_version": 1, "event_id": event_id, "gate_sha256": sha256_file(gate_path),
        "output_dir": str(root), "role": "annotation only; no predictions or scoring",
    })
    root.mkdir(parents=True, exist_ok=False)
    (root / "private").mkdir(mode=0o700)
    records = []
    for region in protocol["regions"]:
        candidate = Path(region["source_candidate"])
        if candidate.parent != Path("data/eval") or not candidate.name.endswith("_gt.las"):
            raise ProtocolViolation("Unknown locked raw-tile naming convention; no guessed data source")
        source = repository / candidate.with_name(candidate.name[:-7] + ".las")
        records.append(_export_region(source, root, region["id"], region["scoring_bounds"]))
    _json(root / "private/package_summary.json", {
        "kind": "holdout", "event_id": event_id, "regions": records,
        "accuracy": None, "status": "awaiting human annotation; do not score or rerun this gate",
    })
    return {"kind": "holdout", "output_dir": str(root), "regions": [
        {"id": r["region_id"], "point_count": r["point_count"]} for r in records],
        "accuracy": None, "labels_created": False, "gate_consumed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["pilot-audit", "holdout"], required=True)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--holdout-sha256", required=True)
    parser.add_argument("--objective-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arrays-path")
    parser.add_argument("--gate-path")
    parser.add_argument("--manifest-path")
    parser.add_argument("--event-id")
    args = parser.parse_args()
    repo = _path(args.repository)
    guard = HoldoutGuard.from_files(repo / "holdout.json", repo / "objective.md",
        expected_holdout_sha256=args.holdout_sha256, expected_objective_sha256=args.objective_sha256)
    if args.kind == "pilot-audit":
        if not args.arrays_path:
            parser.error("pilot-audit requires --arrays-path")
        result = export_pilot_audit(arrays_path=args.arrays_path, raw_tile_path=repo / "data/eval/tile_c.las",
                                    output_dir=args.output_dir, guard=guard)
    else:
        if not all([args.gate_path, args.manifest_path, args.event_id]):
            parser.error("holdout requires --gate-path, --manifest-path and --event-id")
        result = export_holdout_packages(repository=repo, gate_path=args.gate_path,
            manifest_path=args.manifest_path, event_id=args.event_id, output_dir=args.output_dir, guard=guard)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
