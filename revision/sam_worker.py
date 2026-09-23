"""Pinned, offline SAM3 inference for explicitly supplied revision images.

One model is reused across a JSON job list. No historical image or inference
cache is read. The parent experiment runner is responsible for spatial gating.
Importing this module does not import MLX or initialize the GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SAM_ROOT = REPO / "segmentation" / "mlx_sam3"
DEFAULT_PROMPTS = ["road", "sidewalk", "parking lot", "grass", "tree", "building", "car"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_to_rle(mask) -> dict:
    """C-order alternating background/foreground runs, compatible with backend."""
    import numpy as np

    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("Mask must be a nonempty 2D array")
    flat = mask.ravel(order="C")
    changes = np.flatnonzero(np.diff(flat) != 0) + 1
    counts = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"counts": counts, "size": list(mask.shape)}


def load_jobs(args) -> list[dict]:
    if args.jobs:
        if args.image or args.output:
            raise ValueError("Use --jobs OR --image and --output")
        document = json.loads(Path(args.jobs).read_text())
        jobs = document["jobs"] if isinstance(document, dict) else document
    else:
        if not args.image or not args.output:
            raise ValueError("Both --image and --output are required")
        jobs = [{"image": args.image, "output": args.output}]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Jobs must be a nonempty list of image/output objects")
    prepared = []
    outputs = set()
    for item in jobs:
        image, output = Path(item["image"]).resolve(), Path(item["output"]).resolve()
        if image == output:
            raise ValueError("Image and output cannot be the same file")
        if not image.is_file():
            raise FileNotFoundError(image)
        if output.exists() or output in outputs:
            raise FileExistsError(f"Refusing to overwrite or duplicate output: {output}")
        image_digest = sha256(image)
        if item.get("image_sha256") and item["image_sha256"] != image_digest:
            raise ValueError(f"Input image hash mismatch: {image}")
        outputs.add(output)
        prepared.append({"image": image, "output": output, "image_sha256": image_digest})
    return prepared


def safetensor_shapes(path: Path) -> dict:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 64 * 1024 * 1024:
            raise ValueError("Unexpectedly large safetensors header")
        header = json.loads(handle.read(size))
    return {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}


def code_manifest() -> dict:
    paths = sorted((SAM_ROOT / "sam3").rglob("*.py"))
    paths += [Path(__file__), SAM_ROOT / "assets/bpe_simple_vocab_16e6.txt.gz"]
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in paths}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = "UNKNOWN - git revision unavailable"
    versions = {}
    for package in ["mlx", "mlx-metal", "mlx-sam3", "numpy", "Pillow", "huggingface-hub", "ftfy", "regex"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    return {"commit": commit, "source_sha256": hashes, "dependencies": versions, "python": sys.version}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--image")
    parser.add_argument("--output")
    parser.add_argument("--jobs", help="JSON list (or object with jobs) of image/output pairs")
    parser.add_argument("--manifest", help="Parent experiment manifest; parameters and checkpoint identity are authoritative")
    parser.add_argument("--raw-score-floor", type=float, default=0.1)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    args = parser.parse_args(argv)
    parent_manifest = None
    if args.manifest:
        manifest_path = Path(args.manifest).resolve(strict=True)
        parent_manifest = json.loads(manifest_path.read_text())
        expected_model = parent_manifest["model"]
        if Path(args.checkpoint).resolve() != Path(expected_model["checkpoint"]).resolve():
            raise ValueError("Checkpoint path disagrees with parent manifest")
        args.expected_checkpoint_sha256 = expected_model["sha256"]
        args.raw_score_floor = expected_model["raw_score_floor"]
        args.resolution = expected_model["processor_resolution"]
        args.seed = expected_model.get("seed", 0)
        args.prompts = [value["prompt"] for value in parent_manifest["parameters"]["CLASSES"].values()]
    if args.resolution != 1008:
        raise ValueError("This model architecture fixes resolution at 1008; change the architecture explicitly to alter it")
    if not 0 <= args.raw_score_floor <= 1:
        raise ValueError("Raw score floor must lie in [0,1]")
    if not args.prompts or len(set(args.prompts)) != len(args.prompts):
        raise ValueError("Prompts must be nonempty and unique")
    jobs = load_jobs(args)
    if parent_manifest is not None:
        output_root = (REPO / parent_manifest["output_root"]).resolve()
        for job in jobs:
            if not job["image"].is_relative_to(output_root / "slices/tiles") or not job["output"].is_relative_to(output_root / "masks/raw"):
                raise ValueError("Inference job is outside parent manifest's isolated image/raw-response directories")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    digest = sha256(checkpoint)
    if args.expected_checkpoint_sha256 and digest != args.expected_checkpoint_sha256:
        raise ValueError("Checkpoint hash mismatch")
    shapes = safetensor_shapes(checkpoint)
    manifest = code_manifest()
    if parent_manifest is not None:
        manifest["parent_manifest_path"] = str(manifest_path)
        manifest["parent_manifest_sha256"] = sha256(manifest_path)
    manifest.update({
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_tensor_count": len(shapes), "seed": args.seed,
        "backend": "local MLX SAM3 on Metal; explicit checkpoint; network disabled",
        "prompts": args.prompts, "raw_score_floor": args.raw_score_floor,
        "preprocessing": {"mode": "RGB", "resize": [1008, 1008], "resize_filter": "PIL LANCZOS", "scale": "float32 / 255", "normalize": "(pixel - 0.5) / 0.5", "tensor_order": "NCHW"},
        "postprocessing": {"score": "sigmoid(class_logit) * sigmoid(presence_logit)", "score_comparison": "> raw_score_floor", "mask_resize": "bilinear to original image, align_corners=False", "mask_threshold": ">0.5", "rle": "C-order alternating background/foreground", "bbox_filter": "none; parent applies specified filter"},
        "cache_provenance": "Fresh inference from hashed job image; no prior response read",
    })
    # These settings make accidental unpinned downloads fail closed.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(SAM_ROOT))
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import numpy as np
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    mx.set_default_device(mx.gpu)
    device_info = mx.device_info()  # Fail immediately if sandbox has no Metal access.
    manifest["device"] = {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v) for k, v in device_info.items()}
    mx.random.seed(args.seed)
    started = time.perf_counter()
    model = build_sam3_image_model(checkpoint_path=str(checkpoint))
    # The vendored loader uses strict=False. Fail if a parameter remains
    # unmatched instead of silently accepting randomly initialized weights.
    parameters = dict(tree_flatten(model.parameters()))
    absent = set(parameters) - set(shapes)
    # Two buffers are generated deterministically by the inspected source:
    # an upper-triangular causal mask and sinusoidal position-encoding caches.
    # Neither is a learned parameter or a random initialization.
    generated_buffers = sorted(key for key in absent if key.endswith(".attn_mask") or ".position_encoding.cache." in key)
    missing = sorted(absent - set(generated_buffers))
    mismatched = {key: {"model": list(value.shape), "checkpoint": list(shapes[key])} for key, value in parameters.items() if key in shapes and tuple(value.shape) != shapes[key]}
    if missing or mismatched:
        raise RuntimeError(f"Checkpoint is not complete for this model: missing={missing}, shape_mismatches={mismatched}")
    manifest["checkpoint_compatibility"] = {"model_array_count": len(parameters), "missing_learned_parameters": [], "shape_mismatches": {}, "deterministically_generated_buffers": generated_buffers, "checkpoint_extra_keys": sorted(set(shapes) - set(parameters))}
    manifest["model_load_seconds"] = time.perf_counter() - started
    processor = Sam3Processor(model, resolution=args.resolution, confidence_threshold=args.raw_score_floor)
    for job in jobs:
        # Detect modifications after the initial validation as well.
        if sha256(job["image"]) != job["image_sha256"]:
            raise ValueError(f"Input changed before inference: {job['image']}")
        image = Image.open(job["image"]).convert("RGB")
        tick = time.perf_counter()
        state = processor.set_image(image)
        results = {}
        prompt_seconds = {}
        for prompt in args.prompts:
            prompt_start = time.perf_counter()
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt, state)
            mx.eval(state)
            detections = []
            for mask, bbox, score in zip(state["masks"], state["boxes"], state["scores"]):
                mask = np.asarray(mask)
                if mask.ndim == 3:
                    mask = mask[0]
                detections.append({"mask_rle": mask_to_rle(mask > 0.5), "bbox": np.asarray(bbox).tolist(), "score": float(np.asarray(score))})
            results[prompt] = detections
            prompt_seconds[prompt] = time.perf_counter() - prompt_start
        payload = {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "results": results, "width": image.width, "height": image.height,
            "score_floor": args.raw_score_floor,
            "processing_time_ms": (time.perf_counter() - tick) * 1000,
            "prompt_seconds": prompt_seconds,
            "peak_memory_bytes": mx.get_peak_memory(),
            "image_path": str(job["image"]), "image_sha256": job["image_sha256"],
            "run_manifest": manifest,
        }
        job["output"].parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves every previous result.
        with job["output"].open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"output": str(job["output"]), "detections": {key: len(value) for key, value in results.items()}, "seconds": payload["processing_time_ms"] / 1000, "peak_memory_bytes": payload["peak_memory_bytes"]}), flush=True)
        del state
        mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
