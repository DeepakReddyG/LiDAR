"""Fixed, bounded revision experiments; one manifest and isolated process per raster."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import time
from pathlib import Path

import laspy
import numpy as np

from revision import runner as r
from revision.metrics import evaluate_predictions

ROOT = r.ROOT
RASTER_CHANGES = {
    "fixed_anchor": {},
    "grid_025": {"GRID_RESOLUTION": 0.25},
    "grid_1": {"GRID_RESOLUTION": 1.0},
    "rgb_band_075": {"TOP_SURFACE_FT": 0.75},
    "rgb_band_3": {"TOP_SURFACE_FT": 3.0},
}


def validate_scope(m):
    """Check locked boxes and allowed input names before touching data bytes."""
    g = r.guard_for(m)
    lock = g._protocol()
    for field, fixed in [
        ("context_bounds", "allowed_context_bounds"),
        ("score_bounds", "scoring_bounds"),
    ]:
        if m["dataset"][field] != lock["development"][fixed]:
            raise ValueError("Development bounds changed")
        g.assert_development_bounds(m["dataset"][field])
    for key, allowed in [
        ("input", "data/eval/tile_c.las"),
        ("reference", "data/eval/tile_c_gt.las"),
    ]:
        if m["dataset"][key] != allowed:
            raise ValueError("Development input must be preregistered tile C")
    return g


def validate_registered_parameters(m):
    experiment = m["experiment_id"]
    if experiment not in RASTER_CHANGES:
        raise ValueError("Experiment is not in the fixed registered raster plan")
    base_path = (ROOT / m["base_manifest_path"]).resolve()
    if (
        base_path.parent != (ROOT / "revision_work/manifests").resolve()
        or base_path.suffix != ".json"
    ):
        raise ValueError("Base manifest must be a recorded manifest JSON")
    if r.sha(base_path) != m["base_manifest_sha256"]:
        raise ValueError("Recorded base manifest changed")
    base = json.loads(base_path.read_text())
    validate_scope(base)
    expected = copy.deepcopy(base["parameters"])
    expected.update(RASTER_CHANGES[experiment])
    if m["parameters"] != expected:
        raise ValueError(
            "Experiment parameters differ from the registered one-factor change"
        )
    if m["model"] != base["model"]:
        raise ValueError("Model settings changed outside the registered plan")
    if r.sha(ROOT / "revision_work/experiment_plan.md") != m["registered_plan_sha256"]:
        raise ValueError("Registered experiment plan changed")


def validate(m):
    g = validate_scope(m)
    validate_registered_parameters(m)
    if m["source_root"] != "revision_work/fixed_source":
        raise ValueError("Experiments require fixed source snapshot")
    output = (ROOT / m["output_root"]).resolve()
    if output.parent != (ROOT / "revision_work/runs").resolve():
        raise ValueError("Experiment output must be an isolated revision run directory")
    for key in ("input", "reference"):
        if r.sha(ROOT / m["dataset"][key]) != m["dataset"][key + "_sha256"]:
            raise ValueError("Immutable development input changed")
    for p, h in m["source_hashes"].items():
        source = (ROOT / p).resolve()
        allowed_root = any(
            source.is_relative_to(ROOT / directory)
            for directory in (
                "revision",
                "revision_work/fixed_source",
                "segmentation/mlx_sam3/sam3",
            )
        )
        if not ((allowed_root and source.suffix == ".py") or p == r.BPE_PATH):
            raise ValueError(
                "Manifest hashes must identify permitted source/tokenizer files"
            )
        if r.sha(ROOT / p) != h:
            raise ValueError("Experiment source changed: " + p)
    if m["csf_parameters"] != r.actual_csf_parameters(
        m["parameters"]["CLOTH_RESOLUTION"]
    ):
        raise ValueError("CSF parameters changed")
    return g


def make_manifests(base_path):
    from revision.units import validate_coordinate_units

    base_path = Path(base_path).resolve()
    if base_path.parent != (ROOT / "revision_work/manifests").resolve():
        raise ValueError("Base manifest must be a recorded manifest JSON")
    base = json.loads(base_path.read_text())
    validate_scope(base)
    destinations = {
        name: ROOT / f"revision_work/manifests/{name}.json" for name in RASTER_CHANGES
    }
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError(
            "Preserve existing experiment manifests; do not overwrite them"
        )
    if r.sha(ROOT / base["dataset"]["input"]) != base["dataset"]["input_sha256"]:
        raise ValueError("Immutable development input changed")
    with laspy.open(ROOT / base["dataset"]["input"]) as stream:
        units = validate_coordinate_units(stream.header, assumed_units="US survey foot")
    for name, changes in RASTER_CHANGES.items():
        m = copy.deepcopy(base)
        m["phase"] = 2 if name == "fixed_anchor" else 3
        m["source_root"] = "revision_work/fixed_source"
        m["source_policy"] = (
            "Corrected frozen source; no changed references or historical caches"
        )
        m["source_git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        m["source_hashes"] = {
            str(p.relative_to(ROOT)): r.sha(p)
            for p in (ROOT / m["source_root"]).rglob("*.py")
        }
        m["source_hashes"].update(
            {
                k: v
                for k, v in base["source_hashes"].items()
                if k.startswith("segmentation/")
            }
        )
        for name2 in [
            "__init__",
            "guard",
            "metrics",
            "sam_worker",
            "runner",
            "components",
            "experiments",
            "units",
        ]:
            p = ROOT / f"revision/{name2}.py"
            m["source_hashes"][str(p.relative_to(ROOT))] = r.sha(p)
        m["parameters"].update(changes)
        m["base_manifest_path"] = base_path.relative_to(ROOT).as_posix()
        m["base_manifest_sha256"] = r.sha(base_path)
        m["units_validation"] = units
        m["experiment_id"] = name
        m["output_root"] = "revision_work/runs/" + name
        m["variant"] = (
            "registered component matrix; baseline and historical rule variants separately identified"
        )
        m["registered_plan_sha256"] = r.sha(ROOT / "revision_work/experiment_plan.md")
        m["supersedes_failed_attempt"] = None
        with destinations[name].open("x") as stream:
            json.dump(m, stream, indent=2, allow_nan=False)
            stream.write("\n")


def contexts(m, out):
    raw = laspy.read(out / "input.las")
    sel = r.inside(raw, m["dataset"]["score_bounds"])
    idx = np.flatnonzero(sel)
    pilot = r.subset_las(raw, sel)
    x, y, z = map(np.asarray, [pilot.x, pilot.y, pilot.z])
    meta = np.load(out / "slices/grid_meta.npz")
    resolution = float(meta["resolution"])
    rows = np.clip(
        ((float(meta["y_max"]) - y) / resolution).astype(int), 0, int(meta["rows"]) - 1
    )
    cols = np.clip(
        ((x - float(meta["x_min"])) / resolution).astype(int), 0, int(meta["cols"]) - 1
    )
    hag = np.load(out / "derived/hag.npy", mmap_mode="r")[idx]
    exg = np.load(out / "derived/exg.npy", mmap_mode="r")[idx]
    surface = np.load(out / "slices/surface_z.npy")
    return {
        "pilot": pilot,
        "idx": idx,
        "x": x,
        "y": y,
        "z": z,
        "r": rows,
        "c": cols,
        "hag": hag,
        "exg": exg,
        "surface": surface,
    }


def configurations(full=True):
    result = []
    if full:
        result.extend(
            [
                {"id": "geometry_only", "pointwise": "geometry"},
                {"id": "point_rules", "pointwise": "rules"},
            ]
        )
        for source, constraints, smoothing, transfer in itertools.product(
            ["rules", "sam3"], [False, True], [False, True], ["naive", "height_aware"]
        ):
            result.append(
                {
                    "id": f"{source}_c{int(constraints)}_s{int(smoothing)}_{transfer}",
                    "source": source,
                    "constraints": constraints,
                    "smoothing": smoothing,
                    "transfer": transfer,
                }
            )
    else:
        result.append(
            {
                "id": "sam3_c1_s1_height_aware",
                "source": "sam3",
                "constraints": True,
                "smoothing": True,
                "transfer": "height_aware",
            }
        )
    if full:
        anchor = {
            "source": "sam3",
            "constraints": True,
            "smoothing": True,
            "transfer": "height_aware",
        }
        for name, opts in [
            ("tree_height", {"tree_height_check": True}),
            ("physical_smoothing", {"physical_smoothing": True}),
            ("both", {"tree_height_check": True, "physical_smoothing": True}),
        ]:
            result.append(dict(id=name, **anchor, **opts))
        for band in [1.5, 6.0]:
            result.append(dict(id=f"transfer_band_{band:g}", **anchor, band=band))
        for window in [1, 3, 9]:
            result.append(dict(id=f"majority_{window}", **anchor, majority_size=window))
        for bbox in [0.5, 1.0]:
            result.append(dict(id=f"bbox_{bbox:g}", **anchor, bbox=bbox))
    return result


def matrix(m, out, full=True):
    from revision import components as cp

    if full and m["experiment_id"] != "fixed_anchor":
        raise ValueError(
            "The full component matrix is registered only at the fixed anchor"
        )
    if full and not (out / "analysis/metrics.json").is_file():
        raise ValueError(
            "Complete the Phase 2 anchor analysis before the full component matrix"
        )
    analysis = out / ("analysis_full" if full else "analysis")
    analysis.mkdir(exist_ok=False)
    c = r.configure(m, out)
    d = contexts(m, out)
    params = m["parameters"]
    exgg = np.load(out / "slices/exg_grid.npy")
    hagg = np.load(out / "slices/hag_grid.npy")
    void = np.load(out / "slices/void_mask.npy")
    sam = r.load_raw_conf(m, out, c)
    proposals = {"sam3": sam, "rules": cp.rules_confidence(exgg, hagg, params)}
    # Predictions are computed without opening reference labels.
    predictions = []
    for cfg in configurations(full):
        t = time.perf_counter()
        if "pointwise" in cfg:
            labels = (
                cp.geometry_only(d["hag"], params)
                if cfg["pointwise"] == "geometry"
                else cp.classify_rules(d["exg"], d["hag"], params)
            )
            result = {
                "labels": labels,
                "model_score": np.full(len(labels), np.nan, np.float32),
                "prediction_source": np.full(len(labels), 5, np.uint8),
                "routes": np.full(len(labels), "unknown"),
            }
        else:
            conf = proposals[cfg["source"]]
            if "bbox" in cfg:
                sg = r.source_module(m, "segmentation.segment_sam3")
                old = sg.BBOX_FRAC_MAX
                try:
                    sg.BBOX_FRAC_MAX = cfg["bbox"]
                    conf = r.load_raw_conf(m, out, c)
                finally:
                    sg.BBOX_FRAC_MAX = old
            labels, score, stats = cp.fuse_components(
                conf,
                exgg,
                hagg,
                void,
                params,
                **{
                    k: cfg[k]
                    for k in [
                        "constraints",
                        "smoothing",
                        "majority_size",
                        "tree_height_check",
                        "physical_smoothing",
                    ]
                    if k in cfg
                },
            )
            result = cp.transfer_components(
                d["z"].astype(np.float32),
                d["r"],
                d["c"],
                labels,
                score,
                d["surface"],
                d["hag"],
                d["exg"],
                params,
                mode=cfg["transfer"],
                band=cfg.get("band"),
            )
            np.savez_compressed(
                analysis / f"{cfg['id']}_grids.npz", labels=labels, score=score
            )
            r.dump(analysis / f"{cfg['id']}_veto.json", stats)
        result["prediction"] = cp.labels_to_las_codes(result["labels"], params)
        predictions.append((cfg, result, time.perf_counter() - t))
    gt = laspy.read(ROOT / m["dataset"]["reference"])
    gsel = r.inside(gt, m["dataset"]["score_bounds"])
    r.assert_geometry(d["pilot"], gt, gsel)
    ids = np.asarray(d["pilot"].orig_index).copy()
    reference = np.asarray(gt.classification)[gsel].copy()
    boundary = r.reference_boundary(
        d["x"], d["y"], d["z"], reference, m["dataset"]["score_bounds"]
    )
    all_metrics = {}
    for cfg, pred, seconds in predictions:
        strata = r.surface_strata(
            d["z"],
            d["surface"][d["r"], d["c"]],
            cfg.get("band", params["MAP_BACK"]["surface_ft"]),
        )
        strata.update(boundary=boundary, interior=~boundary)
        score = evaluate_predictions(
            ids,
            reference,
            ids,
            pred["prediction"],
            strata=strata,
            routes=pred["routes"],
        )
        score["integrity"]["geometry_checked"] = True
        score["integrity"]["geometry_check"] = (
            "Exact ordered IDs, integer XYZ, header scales and offsets"
        )
        score["configuration"] = cfg
        score["proposal_score_kind"] = (
            "none; pointwise rules"
            if "pointwise" in cfg
            else (
                "binary rule support"
                if cfg["source"] == "rules"
                else "SAM3 detection score; not calibrated probability"
            )
        )
        if "pointwise" in cfg:
            score["route_interpretation"] = (
                "Pointwise method has no pixel transfer/fallback stage; route is unknown/not applicable and source code 5 denotes pointwise rules"
            )
        score["seconds_downstream"] = seconds
        all_metrics[cfg["id"]] = score
        np.savez_compressed(
            analysis / f"{cfg['id']}_predictions.npz",
            ids=ids,
            prediction=pred["prediction"],
            model_score=pred["model_score"],
            prediction_source=pred["prediction_source"],
        )
        if cfg["id"] == "sam3_c1_s1_height_aware":
            mp = r.source_module(m, "reprojection.map_back")
            header = mp.prediction_header(d["pilot"].header)
            records = mp.prediction_records(
                d["pilot"].points,
                header,
                pred["labels"],
                pred["model_score"],
                pred["prediction_source"],
            )
            laspy.LasData(header, records).write(analysis / "pilot_predictions.las")
            # The audit helper reads reference only for reviewed membership, never semantic selection.
            np.savez_compressed(
                analysis / "pilot_arrays.npz",
                ids=ids,
                reference=reference,
                prediction=pred["prediction"],
                baseline=cp.labels_to_las_codes(
                    cp.classify_rules(d["exg"], d["hag"], params), params
                ),
                x=d["x"],
                y=d["y"],
                z=d["z"],
                hag=d["hag"],
                exg=d["exg"],
                r=d["r"],
                c=d["c"],
                surface=strata["surface"],
                boundary=boundary,
                raw_index=d["idx"],
                rgb=np.column_stack(
                    [d["pilot"].red, d["pilot"].green, d["pilot"].blue]
                ),
            )
    r.dump(
        analysis / "metrics.json",
        {
            "development_only": True,
            "manifest_sha256": r.sha(out / "manifest.json"),
            "methods": all_metrics,
        },
    )
    r.dump(
        analysis / "artifact_hashes.json",
        {
            str(p.relative_to(out)): r.sha(p)
            for p in analysis.iterdir()
            if p.name != "artifact_hashes.json"
        },
    )
    for name, a in all_metrics.items():
        print(name, a["error_points"], round(100 * a["primary_loss"], 6), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["manifests", "run", "matrix"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.action == "manifests":
        make_manifests(args.manifest)
        return
    m = json.loads(Path(args.manifest).read_text())
    validate(m)
    out = ROOT / m["output_root"]
    start = time.time()
    if args.action == "run":
        if args.full:
            raise ValueError(
                "Run Phase 2 first without --full, then request matrix --full"
            )
        r.prepare(m, out)
        r.infer(m, out)
    matrix(m, out, args.full)
    r.dump(
        out / ("matrix_full_timing.json" if args.full else "run_timing.json"),
        {"wall_seconds": time.time() - start, "full_component_matrix": args.full},
    )


if __name__ == "__main__":
    main()
