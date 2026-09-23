# M1 SAM3 backend audit — 2026-09-23

Scope: code, local runtime metadata, checkpoint files, and health/device readiness. `holdout.json` was read first. No A/B point cloud or survey-wide image/mask cache was intentionally accessed, no image was decoded/viewed, and no model inference or download was performed by this audit. An initial broad text search beneath the bundled engine emitted embedded notebook base64; it was not decoded, interpreted, or used. Subsequent inspections were restricted to source/configuration files. Root was informed.

## Runtime and availability

- Configured API: `http://localhost:8000`; `/health` returned connection refused. No backend was running at audit time.
- Vendored engine: `segmentation/mlx_sam3/`, tracked inside the parent repository rather than a separate Git worktree. Last parent commit touching it: `685ee66410b3f5ae78e66a15775ec8e4b05e7cfc` (2026-07-27).
- `.venv/bin/python` is currently Python 3.13.15. Its `pyvenv.cfg` creation-version field is 3.13.13, so runtime `sys.version` is authoritative. The historical manual's separate Python 3.14 engine environment is not the current setup.
- Current installed packages: mlx 0.32.0, mlx-metal 0.32.0, mlx-sam3 0.1.0 (editable from the vendored engine), numpy 2.4.6, Pillow 12.2.0, huggingface-hub 1.25.1, ftfy 6.3.1, fastapi 0.140.7, uvicorn 0.51.0, python-multipart 0.0.32, laspy 2.7.0, scipy 1.18.0, opencv-python-headless 5.0.0.93, cloth-simulation-filter 1.1.7, requests 2.34.2, pytest 9.1.1.
- GPU metadata probe inside the filesystem sandbox failed with `[metal::load_device] No Metal device available ... headless, sandboxed, or virtualized`. Import and `metal.is_available()` alone did not detect this failure. Root must perform a scoped unsandboxed device check/run. This is not evidence that the physical M4 has no GPU.
- No package installation or model download is needed for the discovered assets. No credentials were inspected or emitted.

## Pinned checkpoint

`/Users/deepakreddygathpa/.cache/huggingface/hub/models--mlx-community--sam3-image/snapshots/b72a14d8127e17e6f2a3d2e075bbbf4307ba146e/model.safetensors`

- Actual bytes: 3,402,867,661.
- SHA-256: `0ad4c3f42ecf706c4cda63cf58d621699491ed65012b3999284ea370984f7173` (read and verified).
- Hugging Face snapshot: `b72a14d8127e17e6f2a3d2e075bbbf4307ba146e`.
- Tensor header: 1,400 tensors, 1,368 F32 and 32 C64.
- Index SHA-256: `c1054e7da669e64113253922e77aa8006f8aca1434a5ef613193f4fb77df528e`.
- Tokenizer vocabulary SHA-256: `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`.
- Current engine upstream Git commit: UNKNOWN — engine was vendored without its own repository metadata. Parent commit plus per-source hashes identify this revision's executable code. Historical July checkpoint/configuration used for old predictions: UNKNOWN - needs provider/PI or contemporaneous execution record. Existing local availability does not establish historical use.

The default engine loader calls an unpinned Hugging Face `snapshot_download`; the new worker bypasses it with the explicit checkpoint path and offline environment flags.

## Actual inference behavior

- Input converted to RGB, resized to 1008 x 1008 using PIL LANCZOS, converted to float32 /255 and normalized `(x-0.5)/0.5`, NCHW order.
- Text prompts: road, sidewalk, parking lot, grass, tree, building, car.
- Batch endpoint raw detection floor: score > 0.1. The score is sigmoid class logit multiplied by sigmoid presence logit.
- Image backbone runs once per image; each prompt resets prior prompt outputs and reuses the encoded image.
- Instance masks are resized bilinearly to original image size, `align_corners=False`, then thresholded at probability > 0.5.
- Masks serialize in C-order run-length encoding. Box filtering belongs downstream, not in raw inference.
- Existing cache keys use only image filename and prompts, not image/checkpoint/code hashes. These caches are not suitable for revision provenance; fresh isolated outputs are necessary.
- Vendored weight loader uses `strict=False`. The worker separately verifies learned parameter keys and shapes against the safetensors header. It permits only the inspected deterministic causal-mask and sinusoidal-position-cache buffers to be absent from checkpoint weights.

## Bounded fresh-inference route

`revision/sam_worker.py` accepts either `--image IMAGE --output RAW_JSON` or `--jobs JOBS_JSON`, together with `--checkpoint PATH` and optionally `--expected-checkpoint-sha256 HASH`. A jobs document is a list (or object containing `jobs`) of `{image, output, image_sha256?}` objects. The model is loaded once and reused across the list. All inputs/outputs must belong to the parent runner's spatially gated tile-C development workspace until Phase 5. The worker itself is image-only; it cannot infer spatial bounds from a PNG.

Outputs preserve raw detections and record exact input and checkpoint hashes, dependency versions, source hashes, Git commit, seed, preprocessing/postprocessing, peak MLX allocation, and measured timing. Existing outputs are rejected; no historical cache is read. It explicitly chooses GPU execution and fails if Metal is inaccessible. It does not silently switch to a slow CPU path.

The model always encodes at 1008 x 1008, so tiny pilot images do not proportionately reduce neural memory/time. The checkpoint alone occupies about 3.4 GB; runtime peak is UNKNOWN until the first bounded execution. A single tile-C context raster with seven prompts is the initial timing/memory probe. Do not promise a runtime based on old manuscript claims. Multi-job reuse amortizes model loading; if observed extrapolation suggests more than two hours, stop for the user's decision.

## Verification performed

`tests/test_revision_sam_worker.py`: 13 synthetic CPU tests passed (2026-09-23), covering RLE exactness, file/hash identity, no-overwrite behavior, duplicate outputs, safetensors-header extraction and invalid masks. CLI help works without MLX initialization. No inference result is claimed by these tests.

Follow-up from root agent: a scoped escalated Metal device probe succeeded and identified the M4 Pro with an approximately 19 GB recommended GPU working set. The sandbox-only error above is therefore an execution-permission limitation, not a missing physical device. Root will execute the worker outside the sandbox as needed.

The worker also accepts `--manifest`; its checkpoint identity, seven prompts, raw score floor, model resolution and seed are authoritative. With a parent manifest, input images must lie inside that run's `slices/tiles/` and raw responses inside its `masks/raw/`. Parent manifest SHA-256 is retained in each response. Spatial LAS gating is still performed by the parent runner before raster creation.
