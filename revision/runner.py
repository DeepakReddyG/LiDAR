"""Manifest-controlled bounded pipeline. No implicit global caches or holdout reads."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import laspy
import numpy as np

from revision.guard import HoldoutGuard, sha256_file
from revision.metrics import REFERENCE_CODES, evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = "UNKNOWN - needs provider/PI"
DEFAULT_VARIANT = {
    "source": "sam3",
    "constraints": True,
    "smoothing": True,
    "transfer": "height_aware",
    "tree_height_check": False,
    "physical_smoothing": False,
}
_CONFIGURATION_ID = None
PIPELINE_PACKAGES = ("projection", "classification", "reprojection", "segmentation")
BPE_PATH = "segmentation/mlx_sam3/assets/bpe_simple_vocab_16e6.txt.gz"
PARAMETERS = [
    "GRID_RESOLUTION",
    "CHUNK_SIZE",
    "DECIMATE_CELL",
    "CLOTH_RESOLUTION",
    "GROUND_OUTLIER_FT",
    "DTM_GAPFILL_WINDOWS",
    "TOP_SURFACE_FT",
    "TILE_SIZE",
    "TILE_STRIDE",
    "RGB_16BIT_TO_8BIT_DIVISOR",
    "INPAINT_RADIUS_PX",
    "BBOX_FRAC_MAX",
    "MAJORITY_FILTER_SIZE",
    "VETO",
    "FUSE_PRIORITY",
    "MAP_BACK",
    "BASELINE",
    "CLASSES",
    "UNLABELLED_LAS_CODE",
]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return sha256_file(path)


def guard_for(m):
    return HoldoutGuard.from_files(
        ROOT / "holdout.json",
        ROOT / "objective.md",
        expected_holdout_sha256=m["protocol"]["holdout_sha256"],
        expected_objective_sha256=m["protocol"]["objective_sha256"],
    )


def actual_csf_parameters(cloth_resolution):
    """Instantiate configuration only; never run filtering or inspect data."""
    import CSF

    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    return {
        name: getattr(csf.params, name)
        for name in (
            "bSloopSmooth",
            "class_threshold",
            "cloth_resolution",
            "interations",
            "rigidness",
            "time_step",
        )
    }


def assert_source_module(module, source_root):
    """Prevent preloaded modules or namespace fall-through to live/global code."""
    source_root = Path(source_root).resolve()
    locations = (
        [module.__file__]
        if getattr(module, "__file__", None)
        else list(getattr(module, "__path__", []))
    )
    if not locations or any(
        not Path(p).resolve().is_relative_to(source_root) for p in locations
    ):
        raise RuntimeError(
            f"Pipeline module {module.__name__} is outside the frozen source root"
        )


def source_module(m, name):
    module = importlib.import_module(name)
    assert_source_module(module, ROOT / m["source_root"])
    filename = Path(module.__file__).resolve()
    relative = filename.relative_to(ROOT).as_posix()
    if m["source_hashes"].get(relative) != sha(filename):
        raise ValueError(
            f"Imported pipeline module is not covered by manifest hashes: {relative}"
        )
    return module


def make_manifest(path, checkpoint):
    # Protocol expected hashes are taken from already-committed preregistration blobs.
    protocols = {
        n: hashlib.sha256(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"2b0cfaf:{n}.md" if n == "objective" else f"5ce1149:{n}.json",
                ],
                cwd=ROOT,
            )
        ).hexdigest()
        for n in ["holdout", "objective"]
    }
    import config

    lock = json.loads((ROOT / "holdout.json").read_text())
    g = HoldoutGuard.from_files(
        ROOT / "holdout.json",
        ROOT / "objective.md",
        expected_holdout_sha256=protocols["holdout"],
        expected_objective_sha256=protocols["objective"],
    )
    g.assert_development_bounds(lock["development"]["allowed_context_bounds"])
    raw = ROOT / "data/eval/tile_c.las"
    ref = ROOT / "data/eval/tile_c_gt.las"
    with laspy.open(raw) as f:
        h = f.header
        crs = h.parse_crs()
        metadata = {
            "points": h.point_count,
            "point_format": h.point_format.id,
            "las_version": str(h.version),
            "scale": h.scales.tolist(),
            "offset": h.offsets.tolist(),
            "bounds": [h.mins.tolist(), h.maxs.tolist()],
            "crs": str(crs) if crs else UNKNOWN,
        }
    sources = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted((ROOT / "revision_work/source_snapshot").rglob("*.py"))
    }
    sources.update(
        {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / "segmentation/mlx_sam3/sam3").rglob("*.py"))
        }
    )
    sources[BPE_PATH] = sha(ROOT / BPE_PATH)
    sources.update(
        {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / "revision").glob("*.py"))
        }
    )
    deps = {
        x: importlib.metadata.version(x)
        for x in [
            "numpy",
            "laspy",
            "pillow",
            "opencv-python-headless",
            "cloth-simulation-filter",
            "matplotlib",
            "mlx",
            "mlx-metal",
            "mlx-sam3",
            "huggingface-hub",
            "scipy",
        ]
    }
    m = {
        "schema_version": 1,
        "phase": 1,
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_policy": "Exact inherited live pipeline snapshot; code hashes supplement commit. No historical settings inferred.",
        "source_root": "revision_work/source_snapshot",
        "source_hashes": sources,
        "protocol": {
            "holdout_sha256": protocols["holdout"],
            "objective_sha256": protocols["objective"],
        },
        "dataset": {
            "input": "data/eval/tile_c.las",
            "input_sha256": sha(raw),
            "reference": "data/eval/tile_c_gt.las",
            "reference_sha256": sha(ref),
            "context_bounds": lock["development"]["allowed_context_bounds"],
            "score_bounds": lock["development"]["scoring_bounds"],
            "header": metadata,
            "units": "US survey feet - explicit project configuration; source header CRS absent",
            "sensor": UNKNOWN,
            "acquisition_date": UNKNOWN,
            "colorization": UNKNOWN,
            "provider": UNKNOWN,
        },
        "model": {
            "checkpoint": str(Path(checkpoint).absolute()),
            "sha256": sha(checkpoint),
            "repository": "mlx-community/sam3-image",
            "snapshot": "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e",
            "backend": "local MLX code under segmentation/mlx_sam3; dedicated pinned worker",
            "device": "gpu",
            "seed": 0,
            "raw_score_floor": 0.1,
            "processor_resolution": 1008,
            "resize": "PIL LANCZOS RGB to 1008x1008",
            "normalize": "float32 RGB /255 then (x-0.5)/0.5",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": deps,
        },
        "parameters": {k: getattr(config, k) for k in PARAMETERS},
        "csf_parameters": actual_csf_parameters(config.CLOTH_RESOLUTION),
        "cache": {
            "policy": "fresh raw responses in a new run directory; subsequent reuse requires recorded image/checkpoint/code identity",
            "historical_cache_used": False,
        },
        "variant": DEFAULT_VARIANT.copy(),
        "output_root": "revision_work/runs/phase1_fresh",
        "boundary_definition": "Reviewed points within Euclidean 3D distance <=1.0 ft of any reviewed point of another class, or within horizontal distance <=1.0 ft of the pilot perimeter; unreviewed points excluded; evaluation-only, never a predictor",
        "reference_role": "read only after predictions are produced; never preprocessing or inference",
    }
    dump(path, m)
    print(path)


def configure(m, out):
    global _CONFIGURATION_ID
    src = (ROOT / m["source_root"]).resolve()
    identity = json.dumps(
        {
            "source_root": str(src),
            "parameters": m["parameters"],
            "output": str(out.resolve()),
        },
        sort_keys=True,
    )
    if _CONFIGURATION_ID is not None and _CONFIGURATION_ID != identity:
        raise RuntimeError(
            "Only one pipeline configuration per process; start a fresh subprocess"
        )
    for name, module in list(sys.modules.items()):
        if name == "config" or name.split(".")[0] in PIPELINE_PACKAGES:
            assert_source_module(module, src)
    for name in PIPELINE_PACKAGES:
        if name not in sys.modules:
            namespace = types.ModuleType(name)
            namespace.__path__ = [str(src / name)]
            namespace.__package__ = name
            sys.modules[name] = namespace
    sys.path.insert(0, str(src))
    c = source_module(m, "config")
    for k, v in m["parameters"].items():
        setattr(c, k, {int(a): b for a, b in v.items()} if k == "CLASSES" else v)
    for k, p in {
        "LAS_PATH": out / "input.las",
        "DERIVED_DIR": out / "derived",
        "SLICES_DIR": out / "slices",
        "MASKS_DIR": out / "masks",
        "OUTPUT_DIR": out / "output",
        "GRID_META_PATH": out / "slices/grid_meta.npz",
    }.items():
        setattr(c, k, p)
    _CONFIGURATION_ID = identity
    return c


def validate(m):
    g = guard_for(m)
    g.assert_development_bounds(m["dataset"]["context_bounds"])
    g.assert_development_bounds(m["dataset"]["score_bounds"])
    lock = g._protocol()
    for supplied, locked in [
        ("context_bounds", "allowed_context_bounds"),
        ("score_bounds", "scoring_bounds"),
    ]:
        if list(m["dataset"][supplied]) != list(lock["development"][locked]):
            raise ValueError(
                f"Development {supplied} must exactly match preregistration"
            )
    if m.get("variant") != DEFAULT_VARIANT:
        raise ValueError("This runner supports only the frozen Phase 1 default variant")
    if m.get("source_root") != "revision_work/source_snapshot":
        raise ValueError("Phase 1 requires the explicitly frozen source snapshot")
    if m.get("csf_parameters") != actual_csf_parameters(
        m["parameters"]["CLOTH_RESOLUTION"]
    ):
        raise ValueError("CSF parameters differ from the recorded manifest")
    # Dataset whitelist prevents an altered manifest from using A/B via a false box.
    if (
        m["dataset"]["input"] != "data/eval/tile_c.las"
        or m["dataset"]["reference"] != "data/eval/tile_c_gt.las"
    ):
        raise ValueError("Development input must be preregistered tile C")
    for k in ["input", "reference"]:
        if sha(ROOT / m["dataset"][k]) != m["dataset"][k + "_sha256"]:
            raise ValueError("Immutable input changed: " + k)
    for p, h in m["source_hashes"].items():
        if sha(ROOT / p) != h:
            raise ValueError("Source changed: " + p)
    return g


def inside(las, b):
    x, y = np.asarray(las.x), np.asarray(las.y)
    return (x >= b[0]) & (y >= b[1]) & (x < b[2]) & (y < b[3])


def prepare(m, out):
    if out.exists():
        raise FileExistsError("Never overwrite a run; choose a new output_root")
    out.mkdir(parents=True)
    dump(out / "manifest.json", m)
    configure(m, out)
    # Crop only the already permitted C file; no full-survey access.
    with (
        laspy.open(ROOT / m["dataset"]["input"]) as src,
        laspy.open(out / "input.las", mode="w", header=src.header.copy()) as dst,
    ):
        for pts in src.chunk_iterator(500_000):
            keep = inside(pts, m["dataset"]["context_bounds"])
            dst.write_points(pts[keep])
    stages = {}
    for name, mod, fn in [
        ("grid", "projection.grid", "run_grid"),
        ("features", "projection.features", "run_features"),
        ("ground", "projection.ground", "run_ground"),
        ("ortho", "projection.ortho", "run_ortho"),
    ]:
        t = time.time()
        getattr(source_module(m, mod), fn)()
        stages[name] = time.time() - t
    dump(
        out / "preprocessing.json",
        {
            "seconds": stages,
            "input_crop_sha256": sha(out / "input.las"),
            "context": "bounded tile C only; no historical survey features",
            "units": m["dataset"]["units"],
        },
    )
    jobs = [
        {
            "image": str(p.resolve()),
            "image_sha256": sha(p),
            "output": str((out / "masks/raw" / f"{p.stem}.json").resolve()),
        }
        for p in sorted((out / "slices/tiles").glob("*.png"))
    ]
    dump(out / "sam_jobs.json", jobs)
    print("Prepared", len(jobs), "inference images", flush=True)


def infer(m, out):
    checkpoint = m["model"]["checkpoint"]
    if sha(checkpoint) != m["model"]["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "revision.sam_worker",
            "--checkpoint",
            checkpoint,
            "--jobs",
            str(out / "sam_jobs.json"),
            "--manifest",
            str(out / "manifest.json"),
        ],
        cwd=ROOT,
        check=True,
    )


def verified_responses(m, out):
    """Accept only complete, explicitly registered fresh inference responses."""
    from PIL import Image

    saved_manifest = json.loads((out / "manifest.json").read_text())
    if saved_manifest != json.loads(json.dumps(m)):
        raise ValueError("Run manifest differs from requested configuration")
    jobs = json.loads((out / "sam_jobs.json").read_text())
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Expected a nonempty inference job list")
    images = {p.resolve() for p in (out / "slices/tiles").glob("*.png")}
    expected_images, expected_outputs = set(), set()
    prepared = []
    for job in jobs:
        image, output = Path(job["image"]).resolve(), Path(job["output"]).resolve()
        if (
            image.parent != (out / "slices/tiles").resolve()
            or output != (out / "masks/raw" / f"{image.stem}.json").resolve()
        ):
            raise ValueError("Inference job lies outside its isolated run directories")
        if image in expected_images or output in expected_outputs:
            raise ValueError("Duplicate inference job")
        expected_images.add(image)
        expected_outputs.add(output)
        if job.get("image_sha256") != sha(image):
            raise ValueError("Job image provenance does not match current image")
        prepared.append((image, output, job["image_sha256"]))
    if images != expected_images:
        raise ValueError("Inference jobs do not exactly cover generated image tiles")
    outputs = {p.resolve() for p in (out / "masks/raw").glob("*.json")}
    if outputs != expected_outputs:
        raise ValueError("Inference responses are missing or unexpected")
    expected_code = {
        p: h
        for p, h in m["source_hashes"].items()
        if p.startswith("segmentation/mlx_sam3/sam3/")
        or p in {BPE_PATH, "revision/sam_worker.py"}
    }
    if BPE_PATH not in expected_code or "revision/sam_worker.py" not in expected_code:
        raise ValueError("Manifest lacks worker/tokenizer source identities")
    prompts = [i["prompt"] for i in m["parameters"]["CLASSES"].values()]
    parent_hash = sha(out / "manifest.json")
    responses = []
    for image, output, image_hash in prepared:
        response = json.loads(output.read_text())
        provenance = response["run_manifest"]
        if (
            response.get("image_sha256") != image_hash
            or Path(response["image_path"]).resolve() != image
        ):
            raise ValueError("Response image identity differs from its job")
        if provenance.get("parent_manifest_sha256") != parent_hash:
            raise ValueError("Response belongs to a different parent manifest")
        if (
            provenance.get("checkpoint_sha256") != m["model"]["sha256"]
            or Path(provenance["checkpoint_path"]).resolve()
            != Path(m["model"]["checkpoint"]).resolve()
        ):
            raise ValueError("Response checkpoint identity differs from manifest")
        if provenance.get("source_sha256") != expected_code:
            raise ValueError("Response worker code identity differs from manifest")
        if provenance.get("prompts") != prompts or set(response["results"]) != set(
            prompts
        ):
            raise ValueError("Response prompts are incomplete or differ from manifest")
        if provenance.get("raw_score_floor") != m["model"][
            "raw_score_floor"
        ] or provenance.get("seed") != m["model"].get("seed", 0):
            raise ValueError("Response inference parameters differ from manifest")
        if (
            provenance.get("preprocessing", {}).get("resize")
            != [m["model"]["processor_resolution"]] * 2
        ):
            raise ValueError("Response processor resolution differs from manifest")
        with Image.open(image) as picture:
            if (response["width"], response["height"]) != picture.size:
                raise ValueError("Response dimensions differ from the registered image")
        responses.append((output, response))
    return responses


def assert_geometry(pilot, reference, reference_selection):
    for field in ("scales", "offsets"):
        if not np.array_equal(
            getattr(pilot.header, field), getattr(reference.header, field)
        ):
            raise ValueError(f"Prediction/reference header {field} changed")
    ids = np.asarray(pilot.orig_index)
    expected = np.asarray(reference.orig_index)[reference_selection]
    if not np.array_equal(ids, expected):
        raise ValueError(
            "Pilot point ordering differs; exact identity alignment required"
        )
    if not all(
        np.array_equal(
            np.asarray(getattr(pilot, a)),
            np.asarray(getattr(reference, a))[reference_selection],
        )
        for a in ("X", "Y", "Z")
    ):
        raise ValueError("Prediction/reference geometry changed")


def reference_boundary(x, y, z, codes, bounds, distance=1.0):
    """Evaluation-only 3D cross-class proximity plus horizontal crop perimeter."""
    from scipy.spatial import cKDTree

    coordinates = np.column_stack([x, y, z]).astype(np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Boundary geometry must be finite")
    reviewed = np.isin(codes, REFERENCE_CODES)
    perimeter = np.minimum.reduce(
        [x - bounds[0], y - bounds[1], bounds[2] - x, bounds[3] - y]
    )
    boundary = reviewed & (perimeter <= distance)
    for code in np.unique(np.asarray(codes)[reviewed]):
        own = reviewed & (codes == code)
        other = reviewed & (codes != code)
        if other.any():
            tree = cKDTree(coordinates[other])
            nearest, _ = tree.query(coordinates[own], k=1, workers=1)
            boundary[own] |= nearest <= distance
    return boundary


def surface_strata(z, surface, band):
    finite = np.isfinite(surface)
    delta = np.asarray(z, dtype=np.float32) - surface
    on = finite & (np.abs(delta) < band)
    return {
        "surface": on,
        "below_surface": finite & ~on & (delta <= 0),
        "above_surface": finite & ~on & (delta > 0),
        "missing_surface": ~finite,
    }


def load_raw_conf(m, out, c):
    responses = verified_responses(m, out)
    sg = source_module(m, "segmentation.segment_sam3")
    meta = np.load(out / "slices/grid_meta.npz")
    shape = (int(meta["rows"]), int(meta["cols"]))
    conf = {i["name"]: np.zeros(shape, np.float32) for i in c.CLASSES.values()}
    for p, d in responses:
        match = re.fullmatch(r"tile_r(\d+)_c(\d+)", p.stem)
        if match is None:
            raise ValueError("Unexpected tile naming convention")
        r, col = map(int, match.groups())
        if r + d["height"] > shape[0] or col + d["width"] > shape[1]:
            raise ValueError("Response tile extends beyond the configured grid")
        for i in c.CLASSES.values():
            arr = sg.tile_confidence(d["results"][i["prompt"]], d["height"], d["width"])
            np.maximum(
                conf[i["name"]][r : r + arr.shape[0], col : col + arr.shape[1]],
                arr,
                out=conf[i["name"]][r : r + arr.shape[0], col : col + arr.shape[1]],
            )
    return conf


def subset_las(raw, selection):
    """Keep coordinate scaling when laspy slicing returns a packed record."""
    header = raw.header.copy()
    records = laspy.ScaleAwarePointRecord(
        raw.points[selection].array.copy(), header.point_format,
        header.scales, header.offsets,
    )
    return laspy.LasData(header, records)


def score(m, out):
    # Reserve scoring once before imports or writes. Even a failed attempt is
    # preserved; rerun in a new run directory instead of overwriting evidence.
    (out / "output").mkdir(exist_ok=False)
    c = configure(m, out)
    fu = source_module(m, "classification.fuse")
    mp = source_module(m, "reprojection.map_back")
    ba = source_module(m, "classification.baseline")
    conf = load_raw_conf(m, out, c)
    exgg = np.load(out / "slices/exg_grid.npy")
    hagg = np.load(out / "slices/hag_grid.npy")
    void = np.load(out / "slices/void_mask.npy")
    labels, confidence, stats = fu.fuse(conf, exgg, hagg, void)
    raw = laspy.read(out / "input.las")
    sel = inside(raw, m["dataset"]["score_bounds"])
    idx = np.flatnonzero(sel)
    pilot = subset_las(raw, sel)
    meta = np.load(out / "slices/grid_meta.npz")
    x, y, z = map(np.asarray, [pilot.x, pilot.y, pilot.z])
    res = float(meta["resolution"])
    r = np.clip(
        ((float(meta["y_max"]) - y) / res).astype(int), 0, int(meta["rows"]) - 1
    )
    col = np.clip(
        ((x - float(meta["x_min"])) / res).astype(int), 0, int(meta["cols"]) - 1
    )
    hag = np.load(out / "derived/hag.npy", mmap_mode="r")[idx]
    exg = np.load(out / "derived/exg.npy", mmap_mode="r")[idx]
    surf = np.load(out / "slices/surface_z.npy")
    pred, legacy_score = mp.label_points(
        z.astype(np.float32), r, col, labels, confidence, surf, hag, exg
    )
    pred_codes = mp.labels_to_las_codes(pred)
    pilot.classification = pred_codes
    pilot.user_data = legacy_score
    # Only now open the immutable existing reference for scoring.
    gt = laspy.read(ROOT / m["dataset"]["reference"])
    gsel = inside(gt, m["dataset"]["score_bounds"])
    gids = np.asarray(gt.orig_index)[gsel].copy()
    gcodes = np.asarray(gt.classification)[gsel].copy()
    ids = np.asarray(pilot.orig_index).copy()
    assert_geometry(pilot, gt, gsel)
    strata = surface_strata(z, surf[r, col], c.MAP_BACK["surface_ft"])
    on = strata["surface"]
    routes = np.where(on, "transfer", "fallback")
    # Evaluation strata; fixed from reference geometry independently of predictions.
    boundary = reference_boundary(x, y, z, gcodes, m["dataset"]["score_bounds"])
    strata.update({"boundary": boundary, "interior": ~boundary})
    result = evaluate_predictions(
        gids, gcodes, ids, pred_codes, strata=strata, routes=routes
    )
    base = mp.labels_to_las_codes(ba.classify(exg, hag))
    baseline = evaluate_predictions(gids, gcodes, ids, base, strata=strata)
    for metrics in (result, baseline):
        metrics["integrity"]["geometry_checked"] = True
        metrics["integrity"]["geometry_check"] = (
            "Exact ordered IDs, integer XYZ, header scales and offsets"
        )
    # Only validated prediction records are written; references are never changed.
    pilot.write(out / "output/pilot_predictions.las")
    np.save(out / "output/label_grid.npy", labels)
    np.save(out / "output/score_grid.npy", confidence)
    dump(out / "output/veto_stats.json", stats)
    np.savez_compressed(
        out / "output/pilot_arrays.npz",
        ids=ids,
        reference=gcodes,
        prediction=pred_codes,
        baseline=base,
        x=x,
        y=y,
        z=z,
        hag=hag,
        exg=exg,
        r=r,
        c=col,
        surface=on,
        missing_surface=strata["missing_surface"],
        boundary=boundary,
        raw_index=idx,
        rgb=np.column_stack([pilot.red, pilot.green, pilot.blue]),
    )
    dump(
        out / "output/metrics.json",
        {
            "pipeline": result,
            "baseline": baseline,
            "development_only": True,
            "source": "fresh bounded end-to-end pipeline",
            "manifest_sha256": sha(out / "manifest.json"),
        },
    )
    dump(
        out / "output/artifact_hashes.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in (out / "output").glob("*")
            if p.name != "artifact_hashes.json"
        },
    )
    print(
        "Fresh metrics",
        json.dumps(
            {
                "pipeline": result["overall_accuracy"],
                "baseline": baseline["overall_accuracy"],
                "primary_loss": result["primary_loss"],
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["manifest", "prepare", "infer", "score", "all"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint")
    a = p.parse_args()
    if a.action == "manifest":
        make_manifest(a.manifest, a.checkpoint)
        return
    m = json.loads(Path(a.manifest).read_text())
    validate(m)
    out = ROOT / m["output_root"]
    t = time.time()
    if a.action in ["prepare", "all"]:
        prepare(m, out)
    if a.action in ["infer", "all"]:
        infer(m, out)
    if a.action in ["score", "all"]:
        score(m, out)
    print("Elapsed", round(time.time() - t, 2), "seconds", flush=True)


if __name__ == "__main__":
    main()
