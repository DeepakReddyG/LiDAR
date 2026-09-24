"""Read-only independent audit of completed tile-C Phase 1/fixed-anchor runs.

Writes a new evidence JSON only. Does not import or execute model/pipeline code,
alter references, open survey caches, or read any held-out region.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import laspy
import numpy as np

from revision.guard import HoldoutGuard, sha256_file
from revision.units import validate_coordinate_units

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "sam3_c1_s1_height_aware"


def equality(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"same_shape": False, "old_shape": list(a.shape), "new_shape": list(b.shape)}
    same = a == b
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        same |= np.isnan(a) & np.isnan(b)
    record = {"same_shape": True, "shape": list(a.shape), "equal": bool(same.all()), "changed_values": int((~same).sum()), "old_dtype": str(a.dtype), "new_dtype": str(b.dtype)}
    if a.dtype.kind in "iuf" and b.dtype.kind in "iuf":
        finite = np.isfinite(a) & np.isfinite(b)
        record["max_absolute_finite_difference"] = float(np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)))) if finite.any() else None
    return record


def score(reference, prediction):
    """Independent reconstruction of printed metrics without evaluator import."""
    codes = [3, 5, 6, 11, 64]
    keep = ~np.isin(reference, [0, 1])
    reference, prediction = reference[keep], prediction[keep]
    if not len(reference) or not np.isin(reference, codes).all():
        raise ValueError("Unsupported or absent reviewed reference labels")
    cm = np.zeros((5, 7), dtype=np.int64)
    for row, code in enumerate(codes):
        ref = reference == code
        for col, pred_code in enumerate(codes):
            cm[row, col] = np.count_nonzero(ref & (prediction == pred_code))
        cm[row, 5] = np.count_nonzero(ref & (prediction == 1))
        cm[row, 6] = np.count_nonzero(ref & ~np.isin(prediction, [*codes, 1]))
    per_class, losses, ious = {}, [], []
    for i, code in enumerate(codes):
        n = int(cm[i].sum())
        if not n:
            continue
        tp, fp, fn = int(cm[i, i]), int(cm[:, i].sum() - cm[i, i]), int(n - cm[i, i])
        iou = tp / (tp + fp + fn)
        losses.append(fn / n)
        ious.append(iou)
        per_class[str(code)] = {"support": n, "error_points": fn, "iou": iou}
    correct = int(np.trace(cm[:, :5]))
    return {"scored_points": len(reference), "correct_points": correct, "error_points": len(reference) - correct, "unlabelled_points": int(cm[:, 5].sum()), "overall_accuracy": correct / len(reference), "primary_loss": float(np.mean(losses)), "macro_iou": float(np.mean(ious)), "confusion_matrix": cm.tolist(), "per_class": per_class}


def require_metrics_match(actual, saved):
    checks = {}
    for key in ["scored_points", "correct_points", "error_points", "unlabelled_points", "overall_accuracy", "primary_loss", "macro_iou", "confusion_matrix"]:
        a, b = actual[key], saved[key]
        ok = bool(np.isclose(a, b, rtol=0, atol=1e-12)) if isinstance(a, float) else a == b
        checks[key] = ok
    if not all(checks.values()):
        raise ValueError(f"Independent metric reconstruction disagrees: {checks}")
    return checks


def read_manifest(run):
    expected_run = (ROOT / "revision_work/runs" / run.name).resolve()
    if run != expected_run or run.name not in {"phase1_fresh", "fixed_anchor"}:
        raise ValueError("Audit is limited to preregistered Phase 1 and fixed-anchor run directories")
    m = json.loads((run / "manifest.json").read_text())
    guard = HoldoutGuard.from_files(ROOT / "holdout.json", ROOT / "objective.md", expected_holdout_sha256=m["protocol"]["holdout_sha256"], expected_objective_sha256=m["protocol"]["objective_sha256"])
    lock = guard._protocol()
    for name, allowed in [("input", "data/eval/tile_c.las"), ("reference", "data/eval/tile_c_gt.las")]:
        if m["dataset"][name] != allowed:
            raise ValueError("Only tile-C development input/reference are permitted")
    for name, bound_key in [("context_bounds", "allowed_context_bounds"), ("score_bounds", "scoring_bounds")]:
        if m["dataset"][name] != lock["development"][bound_key]:
            raise ValueError("Manifest does not retain the protected development bounds")
        guard.assert_development_bounds(m["dataset"][name])
    return m


def input_provenance(m, run):
    record = {"manifest_sha256": sha256_file(run / "manifest.json"), "protocol": m["protocol"], "source_git_commit": m["source_git_commit"], "source_root": m["source_root"], "model_checkpoint_sha256": m["model"]["sha256"]}
    for name in ["input", "reference"]:
        path = ROOT / m["dataset"][name]
        actual = sha256_file(path)
        if actual != m["dataset"][name + "_sha256"]:
            raise ValueError("Immutable tile-C " + name + " hash changed")
        record[name + "_sha256_verified"] = actual
    source_checks = {}
    for name, digest in m["source_hashes"].items():
        # Only code/text model assets, never an arbitrary data path.
        p = ROOT / name
        allowed = name.startswith(("revision/", "revision_work/source_snapshot/", "revision_work/fixed_source/", "segmentation/mlx_sam3/sam3/")) or name == "segmentation/mlx_sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        if not allowed:
            raise ValueError("Unexpected source hash path: " + name)
        source_checks[name] = p.exists() and sha256_file(p) == digest
    record["current_source_hash_checks"] = source_checks
    record["source_note"] = "Frozen pipeline/model source must remain exact; later orchestration changes are recorded, not mistaken for the old executed code."
    immutable_checks = {k: v for k, v in source_checks.items() if not k.startswith("revision/")}
    if not all(immutable_checks.values()):
        raise ValueError("Frozen pipeline/model source changed")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", default="revision_work/runs/phase1_fresh")
    parser.add_argument("--new-run", default="revision_work/runs/fixed_anchor")
    parser.add_argument("--output", default="revision_work/evidence/fix_audit.json")
    args = parser.parse_args(argv)
    old, new = (ROOT / args.old_run).resolve(), (ROOT / args.new_run).resolve()
    output = (ROOT / args.output).resolve()
    if output.exists():
        raise FileExistsError("Preserve the previous audit; select a new output path")
    if not output.is_relative_to(ROOT / "revision_work/evidence"):
        raise ValueError("Audit output must remain inside revision_work/evidence")
    old_m, new_m = read_manifest(old), read_manifest(new)
    old_metrics = json.loads((old / "output/metrics.json").read_text())
    new_metrics = json.loads((new / "analysis/metrics.json").read_text())
    if old_metrics["manifest_sha256"] != sha256_file(old / "manifest.json") or new_metrics["manifest_sha256"] != sha256_file(new / "manifest.json"):
        raise ValueError("Metrics do not identify the loaded run manifests")
    if old_m["parameters"] != new_m["parameters"] or old_m["model"] != new_m["model"]:
        raise ValueError("The before/after anchor changed parameters or model settings")
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "development_only": True, "holdout_accessed": False, "script_sha256": sha256_file(Path(__file__)), "old_run": str(old.relative_to(ROOT)), "new_run": str(new.relative_to(ROOT)), "provenance": {"old": input_provenance(old_m, old), "new": input_provenance(new_m, new)}}
    result["derived_arrays"] = {}
    for relative in ["derived/dtm.npy", "derived/hag.npy", "derived/exg.npy", "slices/surface_z.npy", "slices/exg_grid.npy", "slices/hag_grid.npy", "slices/void_mask.npy"]:
        a, b = np.load(old / relative, mmap_mode="r"), np.load(new / relative, mmap_mode="r")
        result["derived_arrays"][relative] = equality(a, b)
        del a, b
    old_images = {p.relative_to(old).as_posix(): sha256_file(p) for p in [old / "slices/ortho_rgb.png", *sorted((old / "slices/tiles").glob("*.png"))]}
    new_images = {p.relative_to(new).as_posix(): sha256_file(p) for p in [new / "slices/ortho_rgb.png", *sorted((new / "slices/tiles").glob("*.png"))]}
    result["raster_images"] = {"equal_hash_maps": old_images == new_images, "old": old_images, "new": new_images}
    with np.load(old / "output/pilot_arrays.npz") as a, np.load(new / "analysis/pilot_arrays.npz") as b:
        result["pilot_arrays"] = {name: equality(a[name], b[name]) for name in ["ids", "reference", "prediction", "baseline", "x", "y", "z", "hag", "exg", "r", "c"]}
        if not result["pilot_arrays"]["ids"]["equal"] or not result["pilot_arrays"]["reference"]["equal"]:
            raise ValueError("Pilot IDs or immutable reference values changed")
        before = score(a["reference"], a["prediction"])
        after = score(b["reference"], b["prediction"])
        result["independent_metric_checks"] = {"before": require_metrics_match(before, old_metrics["pipeline"]), "after": require_metrics_match(after, new_metrics["methods"][ANCHOR])}
        result["before_after"] = {"before": before, "after": after, "net_error_change": after["error_points"] - before["error_points"], "primary_loss_change": after["primary_loss"] - before["primary_loss"]}
        ids, expected_prediction = b["ids"].copy(), b["prediction"].copy()
        rows, cols, point_z = b["r"].copy(), b["c"].copy(), b["z"].astype(np.float32)
    with np.load(new / "analysis" / (ANCHOR + "_grids.npz")) as grids:
        model_grid_scores = grids["score"][rows, cols]
        result["pixel_labels"] = equality(np.load(old / "output/label_grid.npy"), grids["labels"])
        result["pixel_scores"] = equality(np.load(old / "output/score_grid.npy"), grids["score"])
    surface = np.load(new / "slices/surface_z.npy")[rows, cols]
    with laspy.open(ROOT / "data/eval/tile_c.las") as reader:
        source_header = reader.header.copy()
        selected = []
        box = new_m["dataset"]["score_bounds"]
        for chunk in reader.chunk_iterator(500_000):
            x, y = np.asarray(chunk.x), np.asarray(chunk.y)
            take = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])
            if take.any():
                selected.append(chunk.array[take].copy())
        source_array = np.concatenate(selected)
    prediction = laspy.read(new / "analysis/pilot_predictions.las")
    old_prediction = laspy.read(old / "output/pilot_predictions.las")
    if not np.array_equal(source_array["orig_index"], ids) or not np.array_equal(np.asarray(prediction.orig_index), ids):
        raise ValueError("Source and prediction ordered pilot IDs differ")
    unchanged = {name: equality(source_array[name], prediction.points.array[name]) for name in source_array.dtype.names if name != "classification"}
    if not all(value["equal"] for value in unchanged.values()):
        raise ValueError("Export changed original point fields besides classification")
    code_map = {int(key): value["las_code"] for key, value in new_m["parameters"]["CLASSES"].items()}
    class_ids = np.asarray(prediction.internal_class_id)
    mapped = np.array([code_map.get(int(value), new_m["parameters"]["UNLABELLED_LAS_CODE"]) for value in class_ids], np.uint8)
    codes = np.asarray(prediction.classification)
    model_scores = np.asarray(prediction.model_score)
    sources = np.asarray(prediction.prediction_source)
    finite_surface = np.isfinite(surface)
    on = np.abs(point_z - surface) < new_m["parameters"]["MAP_BACK"]["surface_ft"]
    expected_source = np.full(len(ids), 2, np.uint8)
    expected_source[~finite_surface] = 3
    expected_source[finite_surface & ~on & (point_z > surface)] = 4
    expected_source[on & (class_ids < 0)] = 0
    expected_source[on & (class_ids >= 0)] = 1
    scored_transfer = sources == 1
    semantics = {"route_mismatch_points": int(np.count_nonzero(sources != expected_source)), "non_transfer_with_non_nan_score": int(np.count_nonzero((~scored_transfer) & ~np.isnan(model_scores))), "transfer_nonfinite_score": int(np.count_nonzero(scored_transfer & ~np.isfinite(model_scores))), "transfer_score_grid_mismatch": int(np.count_nonzero(scored_transfer & (model_scores != model_grid_scores))), "internal_class_mapping_mismatch": int(np.count_nonzero(mapped != codes)), "npz_vs_las_prediction_mismatch": int(np.count_nonzero(codes != expected_prediction))}
    if any(semantics.values()):
        raise ValueError("Prediction export semantics failed: " + str(semantics))
    vlr_checks = []
    for record in source_header.vlrs:
        if record.user_id == "LASF_Spec" and record.record_id == 4:
            continue
        matches = [v for v in prediction.header.vlrs if v.user_id == record.user_id and v.record_id == record.record_id]
        vlr_checks.append({"user_id": record.user_id, "record_id": record.record_id, "preserved": any(v.record_data_bytes() == record.record_data_bytes() for v in matches)})
    result["export"] = {"points": len(ids), "source_field_checks": unchanged, "source_record_bytes": source_header.point_format.size, "new_record_bytes": prediction.header.point_format.size, "added_bytes_per_point": prediction.header.point_format.size - source_header.point_format.size, "source_user_data_overwritten_before": int(np.count_nonzero(source_array["user_data"] != old_prediction.user_data)), "source_user_data_overwritten_after": int(np.count_nonzero(source_array["user_data"] != prediction.user_data)), "header_scales_preserved": bool(np.array_equal(source_header.scales, prediction.header.scales)), "header_offsets_preserved": bool(np.array_equal(source_header.offsets, prediction.header.offsets)), "vlr_checks": vlr_checks, "semantic_anomalies": semantics, "source_counts": {str(code): int(np.count_nonzero(sources == code)) for code in range(5)}, "zero_score_surface_transfers": int(np.count_nonzero(scored_transfer & (model_scores == 0))), "interpretation": "Surface transfer includes labels propagated by smoothing. Model scores are not calibrated confidence. Source user_data is preserved independently."}
    result["units"] = validate_coordinate_units(source_header, assumed_units="US survey foot")
    result["raw_inference"] = {}
    responses = {}
    for tag, run, manifest in [("old", old, old_m), ("new", new, new_m)]:
        records = {}
        expected_manifest_hash = sha256_file(run / "manifest.json")
        expected_code = {name: digest for name, digest in manifest["source_hashes"].items() if name.startswith("segmentation/mlx_sam3/sam3/") or name in {"segmentation/mlx_sam3/assets/bpe_simple_vocab_16e6.txt.gz", "revision/sam_worker.py"}}
        for path in sorted((run / "masks/raw").glob("tile_r*_c*.json")):
            response = json.loads(path.read_text())
            provenance = response["run_manifest"]
            image = run / "slices/tiles" / (path.stem + ".png")
            checks = {
                "parent_manifest": provenance["parent_manifest_sha256"] == expected_manifest_hash,
                "image_hash": response["image_sha256"] == sha256_file(image),
                "image_path": Path(response["image_path"]).resolve() == image.resolve(),
                "checkpoint_hash": provenance["checkpoint_sha256"] == manifest["model"]["sha256"],
                "raw_score_floor": response["score_floor"] == manifest["model"]["raw_score_floor"],
                "source_hashes": provenance["source_sha256"] == expected_code,
            }
            if not all(checks.values()):
                raise ValueError(f"Raw response provenance failed: {path}: {checks}")
            records[path.name] = {"sha256": sha256_file(path), "checks": checks, "prompt_counts": {key: len(value) for key, value in response["results"].items()}}
            responses[(tag, path.name)] = response
        if not records:
            raise ValueError("No raw responses for " + tag)
        result["raw_inference"][tag] = records
    raw_comparison = {}
    old_names, new_names = set(result["raw_inference"]["old"]), set(result["raw_inference"]["new"])
    if old_names != new_names:
        raise ValueError("Different image sets between same-parameter anchor runs")
    for name in sorted(old_names):
        old_response, new_response = responses[("old", name)], responses[("new", name)]
        pair_record = {}
        for prompt in old_response["results"]:
            old_dets, new_dets = old_response["results"][prompt], new_response["results"].get(prompt, [])
            pair_record[prompt] = {"old_count": len(old_dets), "new_count": len(new_dets), "comparison": "same returned order; no assumed instance correspondence if counts/order differ"}
            if len(old_dets) == len(new_dets):
                pair_record[prompt].update({
                    "exactly_equal": old_dets == new_dets,
                    "mask_pairs_different": sum(a["mask_rle"] != b["mask_rle"] for a, b in zip(old_dets, new_dets)),
                    "max_absolute_score_difference": max((abs(a["score"] - b["score"]) for a, b in zip(old_dets, new_dets)), default=0.),
                    "max_absolute_bbox_difference": max((float(np.max(np.abs(np.asarray(a["bbox"]) - np.asarray(b["bbox"])))) for a, b in zip(old_dets, new_dets)), default=0.),
                })
        raw_comparison[name] = pair_record
    result["raw_inference"]["before_after_detection_comparison"] = raw_comparison
    terrain_equal = all(result["derived_arrays"][name]["equal"] for name in ["derived/dtm.npy", "derived/hag.npy"])
    result["fix_assessments"] = {
        "M1_ground_minimum": {
            "before_metrics": "before_after.before", "after_metrics": "before_after.after",
            "observed_dtm_and_hag_identical": terrain_equal,
            "explanation": "The pipeline decimates to one candidate per coarse cell before CSF. fmin corrects the duplicate-cell contract; equal DTM/HAG demonstrates whether this particular pilot was affected. This audit does not invent an accuracy gain from a dormant collision bug.",
            "isolation": "Same input, preprocessing parameters, and model; directly compare DTM/HAG. Fresh model outputs are separately compared for numerical differences.",
        },
        "M_minor_export_provenance": {
            "before_metrics": "before_after.before", "after_metrics": "before_after.after",
            "classification_changed_by_export": semantics["npz_vs_las_prediction_mismatch"],
            "source_user_data_overwritten_before": result["export"]["source_user_data_overwritten_before"],
            "source_user_data_overwritten_after": result["export"]["source_user_data_overwritten_after"],
            "extra_bytes_per_point": result["export"]["added_bytes_per_point"],
            "interpretation": "Metadata repair; evaluate accuracy on classification only. No reference labels changed.",
        },
        "M1_cache_identity": {
            "before_metrics": "before_after.before", "after_metrics": "before_after.after",
            "classification_operation": "none; validates matching response/image/model/code provenance before consumption",
            "checks": "raw_inference.old and raw_inference.new",
            "limitation": "This verifies the new revision path. Legacy filename-only HTTP cache remains unsafe and was not used or opened.",
        },
        "M5_unit_CRS_guard": {
            "before_metrics": "before_after.before", "after_metrics": "before_after.after",
            "classification_operation": "none; explicit metadata validation, no coordinate or threshold conversion",
            "metadata": "units",
            "limitation": "Absent provenance remains UNKNOWN - needs provider/PI; assuming survey-foot units is not verifying source metadata.",
        },
    }
    result["limitations"] = ["Existing reference labels were read only; no independent semantic audit is implied.", "Before/after accuracy is one development patch, not an independent test.", "Cache and unit guards are metadata validation changes, not standalone alternative classifiers; their metric references show unchanged scoring inputs, not a causal performance effect."]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "before_errors": before["error_points"], "after_errors": after["error_points"], "changed_point_predictions": result["pilot_arrays"]["prediction"]["changed_values"], "semantic_anomalies": semantics}, indent=2))
    return result


if __name__ == "__main__":
    main()
