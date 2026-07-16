# strategy.md — AI Implementation Strategy

This file drives the implementation of the LiDAR classification pipeline
(design: `MANUAL.md`). It is written to be handed to an AI coding assistant
**one task at a time, in order**. Each task is a self-contained prompt with
a goal, context, spec, constraints, and a runnable verification gate.

---

## How to use this file

1. Work through tasks **strictly in order** (T0 → T7). Later tasks assume
   earlier artefacts exist. Do not skip ahead to the SAM3 work.
2. Start every session by pasting the **GLOBAL CONTEXT** block below, then
   one task block.
3. A task is finished only when its `/verify` gate passes. If the gate
   fails, fix within the task — do not move on.
4. After each task: commit with the message given in `/commit`.

---

## GLOBAL CONTEXT (paste at the start of every session)

/context
- Repo: LiDAR point cloud classification, branch `exp`. Read `MANUAL.md`
  first — it is the authoritative design. This strategy implements its
  build order (MANUAL §8).
- Pipeline: LAS → RGB nadir ortho → SAM3 text-prompted segmentation →
  physics veto (NDVI/HAG/returns) → Z-aware map-back → labelled LAS,
  scored against hand-labelled ground truth.
- Folder-per-stage layout (MANUAL §4): `projection/` (features, ground,
  ortho), `segmentation/` (mlx_sam3 + segment_sam3), `classification/`
  (fuse), `reprojection/` (map_back), `evaluation/` (evaluate). `data/` is
  the only interface between stages — a stage never imports another stage.
- Dataset: `data/raw/UPark_Merged_PS_NAD83_G18_USFT_las.las` — LAS point
  format 8, 411.5 M points, ~843×1058 ft, units US survey feet. Point
  attributes: XYZ, RGB (16-bit), NIR, intensity, return number/count.
- Two environments, never merged: this repo = python 3.11 venv (`.venv`,
  laspy/numpy/torch-era); `segmentation/mlx_sam3/` = python 3.14 + MLX
  (own venv via uv). Stage 2 crosses that boundary via HTTP.

/constraints (apply to every task)
- NEVER load all 411 M points into RAM. Every per-point pass streams
  `laspy.open(...).chunk_iterator(10_000_000)` and writes to
  `np.lib.format.open_memmap` arrays.
- Per-point arrays are index-aligned with LAS point order. This alignment
  is a contract; any code that reorders points must carry the original
  index along.
- `data/slices/grid_meta.npz` is the ONLY pixel↔world mapping. Never
  recompute row/col ↔ x/y independently.
- All paths, prompts, thresholds, veto rules live in `config.py` at repo
  root. No constants duplicated across modules.
- v1 files (`projection/slice.py`, `segmentation/segment.py`,
  `classification/classify.py`) are reference implementations — reuse
  their binning / SAT-filter / chunked-LAS-write patterns, and delete each
  one only at the task that replaces it (noted per task).
- Keep functions runnable both via `main.py --stage <name>` and as
  `python -m <package>.<module>` for direct testing.
- Each new module gets one smallest-possible runnable check (assert-based
  `__main__` self-check or tiny `test_*.py`) — no test frameworks.
- Distance thresholds are in US survey feet (matching the dataset).

---

## T0 — Evaluation harness (BUILD FIRST)

/goal
Ground truth and a scoring script exist before any classifier does, so
every later change is judged by per-class IoU, not impressions.

/context
Read MANUAL §6.0. Relevant existing code: `projection/slice.py`
(chunked LAS reading pattern), `reprojection/map_back.py` (LAS writing).

/spec
1. `evaluation/crop_tiles.py` — chunked XY-filter crop of 3 tiles
   (~150×150 ft each) from the raw LAS: (a) road+parking, (b) buildings,
   (c) trees over ground. Take tile bounds from `config.py`
   (`EVAL_TILE_BOUNDS: dict[str, tuple[x0, y0, x1, y1]]` — leave
   placeholder bounds with a loud comment; the human picks them by looking
   at the site).
   **Critical:** stamp each cropped point's original global index into the
   output (extra uint32 dimension via laspy `ExtraBytesParams`, name
   `orig_index`). Point order does NOT survive CloudCompare editing; all
   later matching uses this stamp, never file order.
   Output: `data/eval/tile_{a,b,c}.las`.
2. The human hand-labels these in CloudCompare → saves as
   `data/eval/tile_{a,b,c}_gt.las` (classification codes per MANUAL §5
   table). This step is manual — print instructions and exit.
3. `evaluation/evaluate.py` (~60–80 lines) —
   `evaluate(gt_las, pred_las) -> dict`: align points via `orig_index`
   (fallback: match on XYZ rounded to 0.01 ft, with a warning), map LAS
   codes back to class ids (MANUAL §5), print per-class IoU + a confusion
   matrix, return the numbers. Must accept ANY labelled LAS as `pred` —
   it will score both the rule baseline (T3) and the SAM3 pipeline (T6).
4. Create `config.py` in this task: class table (id, name, prompt,
   threshold, LAS code) from MANUAL §5, all `data/` paths, eval tile
   bounds. Later tasks extend it.
5. Wire `--stage evaluate` into `main.py`.

/verify
Self-check in `evaluate.py.__main__`: build two tiny synthetic LAS files
in a temp dir (100 points, known labels, shuffled order but stamped
indices), run `evaluate()`, assert the IoU of a perfect prediction is 1.0
and of a known-corrupted one matches the hand-computed value.

/done-when
`python main.py --stage evaluate --gt data/eval/tile_a_gt.las --pred <any>.las`
prints an IoU table; the synthetic self-check passes.

/commit
`feat(evaluation): eval tiles with index stamping + IoU scoring`

---

## T1 — Per-point spectral features

/goal
NDVI, intensity, and return-count memmaps exist for all 411 M points,
index-aligned with LAS order.

/context
Read MANUAL §6.1 (contains the exact chunked-memmap code pattern).

/spec
`projection/features.py` — one chunked pass over the raw LAS writing three
memmaps to `data/derived/`: `ndvi.npy` (float32, (NIR−R)/(NIR+R+1e-6)),
`intensity.npy` (float32), `nreturns.npy` (uint8, number_of_returns).
Also write `data/derived/ndvi_hist.png` (numpy histogram + matplotlib) —
this is the human gate: the histogram must be bimodal (vegetation vs hard
surface). Print a warning telling the human to inspect it.
Wire `--stage features` into `main.py`.

/verify
Self-check: run the pass on the first chunk only (`--limit-chunks 1` debug
flag), assert output length, dtype, no NaN/inf in NDVI, NDVI within
[−1, 1].

/done-when
All three memmaps exist with `len == header.point_count`; RAM stays flat
during the run (memmap, not accumulation); histogram PNG written.

/commit
`feat(projection): chunked NDVI/intensity/returns feature pass`

---

## T2 — Ground filter → DTM → Height Above Ground

/goal
A per-point `hag.npy` (height above ground) exists — the veto stage and
Z-aware map-back both depend on it.

/context
Read MANUAL §6.2. Dependency: `pip install cloth-simulation-filter` (add
to a `requirements.txt` at repo root; create it if absent). Reuse the
binning pattern from `projection/slice.py` and its `_uniform_filter_2d`
SAT box filter for DTM gap-fill.

/spec
`projection/ground.py`, four steps:
1. Decimate: chunked pass, lowest-Z point per 2×2 ft cell → ~200 k points.
2. CSF on the decimated set (`cloth_resolution` from `config.py`,
   default 2.0) → ground points.
3. DTM grid: min-Z rasterise of ground points at the ortho resolution
   (0.5 ft/px, same `grid_meta` conventions), gap-fill with the SAT box
   filter → `data/derived/dtm.npy`.
4. Per-point HAG: chunked pass, `z − dtm[row, col]` →
   `data/derived/hag.npy` (float32 memmap).
Also write `data/derived/dtm_preview.png`. Human gate: the DTM must look
like smooth bare terrain — embossed buildings/trees mean cloth_resolution
is wrong. Wire `--stage ground`.

/verify
Self-check on synthetic data: a flat plane with one box (building) on it —
assert CSF+DTM recovers the plane elevation under the box within
tolerance, and HAG of box-top points ≈ box height.

/done-when
`hag.npy` exists for all points; DTM preview is bare terrain; HAG is ~0
for ground, positive for structures/canopy.

/commit
`feat(projection): CSF ground filter, DTM grid, per-point HAG`

---

## T3 — Rule-only baseline (no neural net) — measure it

/goal
A complete labelled LAS produced by five numpy comparisons alone, scored
with T0. This is the bar the SAM3 pipeline must beat, and a publishable
comparison either way.

/context
Read MANUAL §6.5 (veto table — used here as the classifier itself) and
§8 step 4.

/spec
`classification/baseline.py` — chunked pass classifying every point
directly from per-point features (no grids, no SAM3):
tree: ndvi > 0.2 or hag > 6 (and nreturns > 1 strengthens it);
grass: ndvi > 0.15 and hag < 2; building: hag > 8 and ndvi ≤ 0.15;
pavement: hag < 1.5 and ndvi ≤ 0.15; else unlabelled. Priority order per
MANUAL §6.5. Thresholds from `config.py`.
Write `data/output/baseline.las` reusing the chunked LAS-write pattern
from `reprojection/map_back.py`. Run T0's evaluate against all three GT
tiles and save the numbers to `data/eval/baseline_scores.json`.
Wire `--stage baseline`.

/verify
The evaluate run itself is the verification. Record per-class IoU in the
JSON; sanity-assert tree and pavement IoU > 0 (rules that produce zeros
are wired wrong).

/done-when
`baseline_scores.json` exists with per-class IoU for tiles a/b/c.

/commit
`feat(classification): rule-only baseline classifier, scored`

---

## T4 — RGB ortho + stat grids + tiles

/goal
The raster SAM3 actually sees: a true-colour nadir ortho that looks like
an aerial photo, plus the stat grids the veto and map-back stages need.

/context
Read MANUAL §6.3. Keep the binning and `grid_meta.npz` format from
`projection/slice.py` EXACTLY (it is the invertible pixel↔point mapping).
This task absorbs `slice.py`; delete it at the end. Dependency:
`opencv-python` (inpainting).

/spec
`projection/ortho.py`, resolution 0.5 ft/px, outputs to `data/slices/`:
- `ortho_rgb.png` — per cell, mean R,G,B of points within 1.5 ft of the
  cell's max Z (top-surface colour). 16-bit → 8-bit via /256.
- `surface_z.npy` — max Z per cell (float32).
- `ndvi_grid.npy`, `hag_grid.npy`, `intensity_grid.npy` — mean per cell.
- `void_mask.npy` — bool, cells with zero points.
- Void fill: `cv2.inpaint` (Telea, radius ~3 px) on the RGB — NOT a box
  blur. The ortho must stay photo-like; that is the entire reason it
  works as SAM3 input.
- `tiles/` — 1024×1024 crops, stride 768 (25 % overlap), filenames
  encoding grid offset (`tile_r{row0}_c{col0}.png`).
- `grid_meta.npz` — x_min, y_min, x_max, y_max, resolution, rows, cols.
All raster passes chunked over the LAS (two passes are acceptable: one
for max-Z, one for the near-surface colour/stat means).
Wire `--stage ortho`. Delete `projection/slice.py` and remove its
references from `main.py`.

/verify
Self-check: round-trip 1000 random points through grid_meta forward
(x,y → row,col) and back, assert within one cell. Assert no remaining
black-void pixels inside the data hull of `ortho_rgb.png`.

/done-when
`ortho_rgb.png` visually resembles an aerial photo (human gate); ~6 tiles
written; all grids share (rows, cols) from grid_meta.

/commit
`feat(projection): true-colour ortho, stat grids, tiling; retire slice.py`

---

## T5 — SAM3 batch endpoint + confidence grids

/goal
Per-class float32 confidence grids for the whole site, produced by SAM3
text prompts across the tiles, through a clean process boundary.

/context
Read MANUAL §6.4 and §2 (design evidence). The backend lives in
`segmentation/mlx_sam3/app/backend/main.py` (FastAPI, sessions,
`Sam3Processor`). Its env is separate (python 3.14 + MLX) — code added
there must not import anything from this repo, and vice versa.
This task replaces `segmentation/segment.py` (SAM2); delete it at the end.

/spec
Part A — in `segmentation/mlx_sam3/app/backend/main.py`, add:
```
POST /segment_batch
  { image: <png base64>, prompts: ["road", "tree", ...] }
→ { "road": [ {mask_rle, score, bbox}, ... ], "tree": [...], ... }
```
Rules (both are non-negotiable, MANUAL §6.4):
1. Return RAW scores — no thresholding server-side. Thresholds are tuned
   downstream without re-running inference.
2. Encode the image ONCE (`processor.set_image`), then iterate prompts on
   the same state with `reset_all_prompts` between them (~7× speedup).
Reuse the existing `mask_to_rle` helper.

Part B — `segmentation/segment_sam3.py` (this repo's env):
1. For each tile in `data/slices/tiles/`, POST to `/segment_batch` with
   all class prompts from `config.py` (backend URL in config,
   default `http://localhost:8000`). Fail fast with a clear message if
   the backend is down (tell the user how to start it).
2. Per class, drop detections whose bbox covers > 80 % of the tile
   (observed whole-image-box failure mode).
3. Union surviving instance masks into a per-tile semantic confidence
   map: per pixel, max score of any covering mask.
4. Stitch tiles into site-wide grids (max over the 25 % overlaps) →
   `data/masks/conf_<class>.npy`, one `(rows, cols) float32` per class.
Cache raw per-tile responses to `data/masks/raw/` so re-stitching does
not re-run inference. Wire `--stage segment`. Delete
`segmentation/segment.py`.

/verify
Part A: with the backend running, curl `/segment_batch` on one of the
mlx_sam3 sample images (e.g. `IMG_5.png`) with prompts ["car"], assert a
non-empty result with scores in (0, 1].
Part B self-check: stitch two synthetic overlapping tiles with known
values, assert max-merge output.

/done-when
`conf_<class>.npy` exists for every class in the §5 table, aligned with
grid_meta dims; runtime per tile is ~constant in prompt count (embedding
reuse working).

/commit
`feat(segmentation): SAM3 /segment_batch + tiled confidence grids; retire SAM2`

---

## T6 — Fuse (physics veto) + Z-aware map-back

/goal
The full pipeline output: `labelled.las` where surface points carry
SAM3-derived labels, below-surface points carry LiDAR-rule labels, and
every point carries a confidence value.

/context
Read MANUAL §6.5, §6.6, §7. Modify `reprojection/map_back.py` in place
(keep its grid indexing + chunked LAS writing). This task replaces
`classification/classify.py`; delete it at the end.

/spec
Part A — `classification/fuse.py`:
1. Threshold each `conf_<class>.npy` with the per-class threshold from
   `config.py`.
2. Physics veto per MANUAL §6.5 table (NDVI/HAG grids, void_mask). Log
   the veto-rejection rate per class to stdout and
   `data/masks/veto_stats.json`.
3. Priority painting, first claim wins: vehicle → tree → building →
   sidewalk → parking → road → grass.
4. Majority filter (5×5 mode) over the painted grid to kill speckle.
5. Outputs: `data/masks/label_grid.npy` ((H,W) int32, −1 unlabelled) and
   `data/masks/conf_grid.npy` (winning confidence per pixel, float32).

Part B — `reprojection/map_back.py` (rewrite the core loop, chunked):
```
surf       = surface_z[row, col]
on_surface = |z − surf| < 3.0 ft
labels[on_surface] = label_grid[row, col]
below-surface: hag > 2 → tree; hag ≤ 2 & ndvi > 0.15 → grass;
               hag ≤ 2 & ndvi ≤ 0.15 → road/pavement
```
Write `data/output/labelled.las`: `classification` = LAS codes from the
§5 table; `user_data` = confidence 0–255 (winning pixel confidence for
surface points, fixed 128 for rule-labelled below-surface points).
Also write `data/output/labelled_points.npz` (X, Y, Z, label).
Wire `--stage fuse` and keep `--stage map_back`. Delete
`classification/classify.py`.

/verify
Fuse self-check: synthetic 20×20 grids with known confidences and veto
features, assert painted labels and one speckle pixel removed by the
majority filter. Map-back self-check: synthetic column of points over one
labelled pixel (ground + canopy heights), assert surface point gets the
pixel label and below-surface points get rule labels.
Then the real gate: `python main.py --stage evaluate` against all three
GT tiles → save `data/eval/sam3_scores.json`.

/done-when
`labelled.las` opens in CloudCompare and looks sane (human gate); under a
tree, canopy = tree and ground = grass/pavement (spot-check); IoU numbers
recorded next to the baseline's.

/commit
`feat(pipeline): physics-veto fuse + Z-aware map-back with confidence`

---

## T7 — Tune against the eval bar (ongoing)

/goal
The SAM3 pipeline beats the T3 rule-only baseline on per-class IoU.

/context
Everything is now tunable without re-running inference (raw scores cached
in `data/masks/raw/`, thresholds applied in fuse): iterate on prompts
(needs re-inference), thresholds, veto rules, priority order.

/spec
Loop: change ONE thing in `config.py` → `--stage fuse` → `--stage
map_back` → `--stage evaluate` → record. Use `veto_stats.json` to decide
whether SAM3 or the thresholds are the weak link (high rejection rate =
threshold/veto problem; low rejection + low IoU = prompt/model problem).
Keep a running log in `data/eval/tuning_log.md` (change, per-class IoU,
verdict).
Realistic bars (MANUAL §6.0): pavement & tree > 0.8, building > 0.7,
grass > 0.6; vehicle is the hardest.

/done-when
SAM3 pipeline ≥ baseline on every class it targets, and the gap is
recorded — that comparison is the write-up result either way.

/commit
per tuning milestone: `tune: <what changed> — <IoU delta>`

---

## Standing orders for the AI (any task)

- If a spec conflicts with `MANUAL.md`, MANUAL.md wins — flag the
  conflict, don't silently pick.
- If an input artefact is missing, name the task that produces it and
  stop; never regenerate another stage's output ad hoc.
- Prefer the patterns already in the repo (chunked laspy iteration, SAT
  filter, memmaps, `--stage` CLI) over new abstractions.
- No new dependency beyond: laspy, numpy, matplotlib, opencv-python,
  cloth-simulation-filter, requests (this repo) — anything else needs an
  explicit human yes.
- Human gates (NDVI histogram, DTM preview, ortho appearance,
  CloudCompare inspection) cannot be auto-passed: produce the artefact,
  print what to look for, and wait.
