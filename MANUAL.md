# LiDAR Point Cloud Classification — Project Manual

**Branch:** `exp` · **Status:** v2 pipeline built and runnable (all 6 stages, T0–T6 shipped); v1 files removed, ongoing work is tuning against the eval bar (T7, §9)

This manual is the single authoritative description of the project: what the
finished system is, how it is structured, how every stage works in detail,
how stages hand data to each other, how the result is measured, and in what
order to build it. It supersedes the v1 pipeline (SAM2 automatic masks +
untrained classifier), whose post-mortem is preserved in the Appendix.

---

## 1. The finished system

You drop a raw `.las` survey into `data/raw/`, run one command with a set of
plain-English class prompts, and get back the same point cloud with every one
of its ~411 million points carrying a class label and a confidence value —
viewable in CloudCompare or Potree as a colour-coded 3D scene, and scored
against hand-labelled ground truth so every claim about accuracy is a number,
not an impression.

The pipeline in one line:

> LAS point cloud → true-colour nadir ortho → **SAM3 text-prompted
> segmentation** (the `mlx_sam3` port) → **physics veto** with LiDAR-derived
> features → **Z-aware map-back** → labelled LAS, scored against
> hand-labelled tiles.

Three ideas carry the design:

1. **Evaluation first.** Ground-truth tiles and `evaluate.py` are built
   before any classifier exists. Every prompt, threshold, and veto change is
   judged by per-class IoU and nothing else.
2. **Appearance + physics.** SAM3 says what things *look like*; the LiDAR
   attributes (ExG, height-above-ground, return counts) say what they *are*.
   Each vetoes the other's mistakes.
3. **Z-aware map-back.** 2D labels apply only to points near the visible
   surface. Points below the surface (under canopy, under eaves) are
   classified by LiDAR rules. Under a tree: canopy = tree, trunk = tree,
   ground = grass or pavement.

---

## 2. Design evidence — what the SAM3 experiments established

The `mlx_sam3` experiments are the empirical basis. These are rules, not
preferences:

| Finding | Evidence | Design rule |
|---|---|---|
| Nadir ortho works; oblique fails | oblique @0.5: "road" = 2 objects incl. a whole-image box; nadir @0.3: clean road network, 13 parking lots | **Segment only the top-down orthographic raster.** Never oblique renders. |
| Amorphous surfaces need low thresholds; objects don't | road/parking useless @0.5, good @0.3; cars/trees/buildings fine @0.5 | **Per-class thresholds**, not one global value. |
| Whole-image boxes are a known failure mode | oblique "road" 0.60 box spanning the frame | Drop detections whose box covers > 80 % of the tile. |
| SAM3 handles colourised point-cloud rasters | Temple-Texas ortho segmented well despite scan-line voids | An RGB ortho from the LAS is a valid model input. **Fill the black voids first** — they are the one artefact a photo-trained model never saw. |
| Text prompts return *labelled* masks | every panel | Segmentation and classification are **one stage**. No per-mask classifier exists in this pipeline. |

---

## 3. Dataset facts that shape the design

`UPark_Merged_PS_NAD83_G18_USFT_las.las` — **LAS point format 8**, 411.5 M
points, ~843 × 1058 ft site, ~460 pts/ft², all currently unclassified. Every
point carries:

- **RGB (16-bit)** → the true-colour ortho SAM3 sees, and the vegetation
  proxy: **ExG = (2G − R − B)/(R + G + B)** — empirically bimodal on this
  data (hard-surface spike at 0, vegetation bump ≈0.1–0.3)
- **NIR — present in the format but entirely zero** (verified 2026-07-16:
  min = max = 0 across the file). NDVI is impossible on this dataset; every
  vegetation rule uses ExG instead.
- **Intensity** (17k–61k) → asphalt vs concrete/grass separation
- **Return number / count (1–5)** → multiple returns = canopy, but **weak
  on this data**: only ~1.2 % of points are multi-return (sampled) — a
  hint, never a primary veto
- Units are **US survey feet** — all distance thresholds in this manual are
  in feet. *(Enhancement: read the unit/CRS from the LAS header at runtime
  rather than hardcoding; the thresholds then scale by the unit factor.)*

v1 read only XYZ. The spectral attributes are what make the veto stage and
under-canopy labelling possible.

Practical consequence of the size: **nothing ever loads the full point list
into RAM.** Every per-point pass streams `laspy.chunk_iterator` (10 M points
per chunk) and writes results to memory-mapped `.npy` arrays that are
index-aligned with LAS point order. That index alignment is the contract the
whole pipeline rests on.

---

## 4. Repository layout

One folder per stage; `data/` is the only interface between stages — stage
N+1 never imports stage N.

```
LiDAR/
├── main.py                    # orchestrator: --stage CLI, the only file that
│                              #   knows stage order
├── config.py                  # every path, prompt, threshold, veto rule —
│                              #   no constants scattered across modules
│
├── projection/                # STAGE 1 — LAS → rasters & per-point features
│   ├── features.py            #   ExG / intensity / returns (memmaps)
│   ├── ground.py              #   CSF ground filter → DTM → per-point HAG
│   ├── ortho.py               #   RGB nadir ortho + stat grids + tiles
│   └── slice.py               #   [v1 reference: binning + SAT filter live here
│                              #    until absorbed; then deleted]
│
├── segmentation/              # STAGE 2 — rasters → per-class confidence grids
│   ├── mlx_sam3/              #   the SAM3 engine (own env: python 3.14 + MLX)
│   ├── segment_sam3.py        #   drives mlx_sam3 across tiles, stitches grids
│   └── segment.py             #   [v1 reference: SAM2 AMG; deleted at step 6]
│
├── classification/            # STAGE 3 — confidence grids → label grid
│   ├── fuse.py                #   thresholds + physics veto + priority painting
│   └── classify.py            #   [v1 reference: deleted at step 7]
│
├── reprojection/              # STAGE 4 — label grid → labelled point cloud
│   └── map_back.py            #   Z-aware, chunked, writes labelled.las
│
├── evaluation/                # STAGE 0 — built FIRST
│   └── evaluate.py            #   per-class IoU + confusion matrix vs GT tiles
│
├── data/                      # the handshake between stages
│   ├── raw/                   #   input .las (never in git)
│   ├── derived/               #   per-point memmaps: exg, hag, intensity, …
│   ├── slices/                #   ortho, stat grids, grid_meta, tiles/
│   ├── masks/                 #   per-class confidence grids, label_grid
│   ├── eval/                  #   hand-labelled GT tiles
│   └── output/                #   labelled.las, labelled_points.npz
│
└── models/                    # checkpoints (not in git)
```

Environment note: this repo runs **python 3.11 + torch-era deps**;
`segmentation/mlx_sam3/` runs **python 3.14 + MLX**. They are never merged —
stage 2 crosses a process boundary (HTTP to the FastAPI backend, or a
subprocess CLI). Same contract either way; pick one and stop thinking about
it.

---

## 5. Class set, prompts, thresholds

Coarse classes first (fits a campus/park site). Guardrail/curb are postponed
— ~1 px at this resolution; revisit only after coarse IoU is good.

| id | class | SAM3 prompt | start threshold | LAS code |
|---|---|---|---|---|
| 0 | pavement/road | `"road"` | 0.30 | 11 (road surface) |
| 1 | sidewalk/path | `"sidewalk"` | 0.30 | 11 |
| 2 | parking lot | `"parking lot"` | 0.30 | 11 |
| 3 | grass | `"grass"` | 0.35 | 3 (low veg) |
| 4 | tree | `"tree"` | 0.45 | 5 (high veg) |
| 5 | building | `"building"` | 0.50 | 6 (building) |
| 6 | vehicle | `"car"` | 0.50 | 64 (custom) |
| −1 | unlabelled | — | — | 1 (unassigned) |

Thresholds start from the 0.3-vs-0.5 experiments and are tuned **only**
against `evaluate.py` numbers. They live in `config.py` and are applied in
`fuse.py` — never inside the SAM3 backend (see §6.4 for why).

---

## 6. Stage-by-stage specification

### 6.0 `evaluation/evaluate.py` — build FIRST (≈1 day)

"Not effective" must become a number before any threshold is tuned.

1. **Crop 3 tiles** (~150 × 150 ft each, a chunked laspy XY filter):
   (a) road + parking, (b) buildings, (c) trees over ground. A few million
   points each.
2. **Stamp provenance before labelling.** Write each cropped point's original
   index into an extra byte/int field (or keep a sidecar `orig_index.npy`).
   Point order is **not** guaranteed to survive CloudCompare's
   segment-and-save cycle, and silent misalignment would poison every IoU
   number the entire tuning loop depends on. Match on the stamped index (or,
   as a fallback, on rounded XYZ), never on file order.
3. **Hand-label in CloudCompare** (segment tool → classification code), 2–3 h
   total → `data/eval/tile_{a,b,c}_gt.las`.
4. **`evaluate.py`** (~60–80 lines): load GT tile + any labelled LAS, align
   by stamped index, print per-class IoU and a confusion matrix.

Realistic bars on this data: pavement & tree IoU > 0.8, building > 0.7,
grass > 0.6; vehicle is the hardest. Every change to a prompt, threshold, or
veto is judged by this script and nothing else.

### 6.1 `projection/features.py` — per-point spectral features (≈½ day)

One chunked pass, no dependencies beyond laspy/numpy:

```python
with laspy.open(LAS_PATH) as f:
    n_pts = f.header.point_count
    exg = np.lib.format.open_memmap("data/derived/exg.npy", mode="w+",
                                     dtype=np.float32, shape=(n_pts,))
    # same pattern for intensity.npy (float32), nreturns.npy (uint8)
    off = 0
    for ch in f.chunk_iterator(10_000_000):
        r, g, b = (np.asarray(getattr(ch, c), np.float32) for c in ("red", "green", "blue"))
        exg[off:off + len(ch)] = (2*g - r - b) / (r + g + b + 1e-6)
        ...
        off += len(ch)
```

Outputs (`data/derived/`, all index-aligned with LAS point order):
`exg.npy`, `intensity.npy`, `nreturns.npy`. ~1.6 GB per float32 array on
disk; memory-mapped, so RAM stays flat.

**Gate before proceeding:** plot the ExG histogram. It must show the
hard-surface spike at 0 plus a vegetation bump ≈0.1–0.3. *(Gate outcome,
2026-07-16: the original NDVI plan failed this gate — NIR is all zeros in
this dataset — and ExG passed it on a 5 M-point sample. That is why this
stage computes ExG.)*

### 6.2 `projection/ground.py` — DTM + Height Above Ground (≈1 day)

`pip install cloth-simulation-filter` (CSF). 411 M points won't fit CSF —
they don't need to:

1. **Decimate:** lowest-Z point per 2 × 2 ft cell (a chunked pass reusing the
   v1 binning pattern) → ~200 k candidate points.
2. **CSF** on the decimated set (`cloth_resolution ≈ 2.0` ft to start) →
   ground points.
3. **DTM grid:** rasterise ground points (min-Z per cell), gap-fill with the
   v1 SAT box filter (`_uniform_filter_2d` — keep it, it's correct).
4. **Per-point HAG:** `hag.npy` memmap = `z − dtm[row, col]`, one chunked
   pass.

Outputs: `data/derived/dtm.npy`, `data/derived/hag.npy`.

**Gate:** render the DTM as an image. Smooth terrain only — any embossed
building or tree outline means `cloth_resolution` is off.

### 6.3 `projection/ortho.py` — the raster SAM3 actually sees (≈½–1 day)

Keep v1's binning and `grid_meta.npz` **exactly** (that code is correct — it
is the invertible forward/backward pixel↔point mapping the whole pipeline
pivots on). Change what fills the cells. Resolution stays 0.5 ft/px.

Outputs (`data/slices/`):

- **`ortho_rgb.png`** — per cell, mean R,G,B of points within 1.5 ft of the
  cell's max Z (top-surface colour, not colour smeared through canopy).
  16-bit → 8-bit: divide by 257 (65535 / 255, exact — not the 256 shortcut).
- **`surface_z.npy`** — max Z per cell. This drives the Z-aware map-back.
- **`exg_grid.npy`, `hag_grid.npy`, `intensity_grid.npy`** — mean per cell,
  for the veto stage.
- **`void_mask.npy`** — cells containing zero points (the black scan lines
  seen in the Temple-Texas render). Any label on a void pixel is vetoed at
  fuse time.
- **Void fill:** fill void RGB with **`cv2.inpaint` (Telea)**, not a box
  blur. The entire reason for the RGB ortho is being in-distribution for a
  photo-trained model; inpainting produces photo-like texture, a box-filter
  mean produces smears no photograph contains. Same one line of code.
- **`tiles/`** — 1024 × 1024 px crops, stride 768 (25 % overlap), saved with
  their grid offsets in the filename or a sidecar. At 0.5 ft/px the site is
  ~1700 × 2100 px → ~6 tiles. Tiling matters because a car is ~30 × 15 px;
  downscaling the whole ortho into SAM3's input resolution would shrink it
  below detectability.
- `grid_meta.npz` — x_min, y_min, resolution, rows, cols (v1 format,
  unchanged).

### 6.4 `segmentation/segment_sam3.py` + the mlx_sam3 endpoint (≈1–2 days)

**The process boundary.** Add one headless endpoint to the existing FastAPI
backend (`segmentation/mlx_sam3/app/backend/main.py`):

```
POST /segment_batch
  { image: <png base64>, prompts: ["road", "tree", ...] }
→ { "road": [ {mask_rle, score, bbox}, ... ], "tree": [...], ... }
```

Two rules for this endpoint:

1. **Return raw scores; do NOT threshold in the backend.** Thresholds are
   applied in `fuse.py`. If the backend thresholds, every tuning tweak
   re-runs neural inference; if it returns raw confidence, tuning is a
   seconds-long numpy loop. You will tune dozens of times — this is the
   difference between an afternoon and a week.
2. **Encode the image once per tile, iterate prompts on the same state.**
   `Sam3Processor` already supports `set_image()` once followed by multiple
   `set_text_prompt()` calls (with `reset_all_prompts()` between them). The
   image encoder is the expensive part; with 7 classes this is a ~7×
   inference speedup over re-encoding per prompt.

(If HTTP annoys you, a CLI in the mlx_sam3 tree — tile dir in, `.npz` of
masks + scores out, driven via `subprocess` — is equally fine. Same contract,
pick one.)

On this side, `segment_sam3.py`, per tile and per class:

1. Drop detections whose bbox covers > 80 % of the tile (the observed
   whole-image-box failure mode).
2. Union all surviving instance masks into a **semantic confidence grid**:
   per pixel, the max score of any covering mask. Instance identity is noise
   at this stage — coarse civil classes need per-pixel semantics, and
   semantic union makes tile stitching trivial (max over overlaps) instead
   of an instance-matching problem.
3. Stitch tiles into one `(H, W) float32` confidence grid per class →
   `data/masks/conf_<class>.npy`.

### 6.5 `classification/fuse.py` — physics veto + priority painting (≈½ day)

SAM3 says what things look like; the LiDAR knows what they are. Apply the
per-class thresholds (from `config.py`) to the confidence grids, then veto
with the §6.3 stat grids. Starting rules — calibrate on eval tiles:

| class | keep a pixel only if |
|---|---|
| tree | ExG > 0.10 **or** HAG > 6 |
| grass | ExG > 0.05 **and** HAG < 2 |
| building | HAG > 8 |
| road / sidewalk / parking | HAG < 1.5 |
| vehicle | 1 < HAG < 9 |
| any | not on `void_mask` |

(ExG thresholds are first estimates from the sample histogram — the
hard-surface spike sits at 0, vegetation from ≈0.05–0.1 up. Calibrate on
eval tiles.)

Then paint `label_grid.npy` most-specific-first, first claim wins:
**vehicle → tree → building → sidewalk → parking → pavement → grass**
(config.py class name is "pavement"; its SAM3 prompt text is "road").

Post-paint cleanup: one **majority filter** pass (5×5 mode filter,
`config.MAJORITY_FILTER_SIZE`)
over `label_grid` to kill single-pixel speckle before it becomes thousands
of mislabelled 3D points.

**Log the veto-rejection rate per class** — it is a free diagnostic of
whether SAM3 or the thresholds are the weak link.

Outputs: `data/masks/label_grid.npy` (`(H, W) int32`, −1 = unlabelled),
`data/masks/conf_grid.npy` (winning confidence per pixel, for §6.6).

### 6.6 `reprojection/map_back.py` — Z-aware, chunked (≈1 day)

Keep v1's grid indexing and chunked LAS writing; fix the column bug. Per
10 M-point chunk:

```python
row, col   = <v1 indexing from grid_meta>
surf       = surface_z[row, col]
on_surface = np.abs(z - surf) < 3.0                     # ft
labels[on_surface] = label_grid[row, col][on_surface]

# below-surface points (under canopy, under eaves): LiDAR rules, not SAM3
below = ~on_surface
labels[below & (hag > 2)]                  = TREE       # trunk / understory
labels[below & (hag <= 2) & (exg > 0.05)]  = GRASS
labels[below & (hag <= 2) & (exg <= 0.05)] = ROAD_OR_PAVEMENT
```

This is the payoff of the whole design: v1 structurally could not label a
tree's trunk and the ground beneath it differently from its canopy.

Write to `data/output/labelled.las`, chunked:

- `classification` — the class → LAS-code mapping from the §5 table.
- **`user_data` — per-point confidence scaled to 0–255** (the winning
  pixel's confidence; 0 for rule-labelled below-surface points or define a
  fixed rule-confidence, e.g. 128). Free at write time, and it lets any
  downstream consumer filter low-confidence points instead of trusting
  labels blindly.

Also write `data/output/labelled_points.npz` (X, Y, Z, label) for quick
numpy consumers.

### 6.7 `main.py` + `config.py` — orchestration

Keep the `--stage` CLI shape. Stages: `features`, `ground`, `ortho`,
`segment`, `fuse`, `map_back`, `evaluate` (+ `all`). Each stage checks its
input files exist and fails with a message naming the stage that produces
them. `config.py` holds every path, prompt, threshold, and veto rule — v1
duplicated paths across four files; never again.

---

## 7. Data contracts (the full handshake table)

| artefact | producer | consumers | shape / format |
|---|---|---|---|
| `data/derived/exg.npy`, `intensity.npy`, `nreturns.npy` | features.py | fuse (via grids), map_back | `(N,)` memmap, LAS point order |
| `data/derived/dtm.npy`, `hag.npy` | ground.py | ortho, fuse, map_back | grid / `(N,)` memmap |
| `data/slices/ortho_rgb.png` + `tiles/` | ortho.py | segment_sam3 | uint8 RGB |
| `data/slices/surface_z.npy` | ortho.py | map_back | `(H, W)` float32 |
| `data/slices/{exg,hag,intensity}_grid.npy`, `void_mask.npy` | ortho.py | fuse | `(H, W)` |
| `data/slices/grid_meta.npz` | ortho.py | segment_sam3, map_back, evaluate | x_min, y_min, resolution, rows, cols |
| `data/masks/conf_<class>.npy` | segment_sam3.py | fuse | `(H, W)` float32, raw scores |
| `data/masks/label_grid.npy`, `conf_grid.npy` | fuse.py | map_back | `(H, W)` int32 / float32 |
| `data/output/labelled.las` | map_back.py | evaluate, CloudCompare/Potree | LAS + classification + user_data |
| `data/eval/tile_*_gt.las` | hand labelling | evaluate | LAS + stamped original index |

Two invariants:

1. **Per-point arrays are index-aligned with LAS point order.** Any pass
   that reorders points must carry the index along.
2. **`grid_meta.npz` is the only pixel↔world mapping.** No module computes
   its own; row/col ↔ x/y goes through it in both directions.

---

## 8. Build order

| # | Task | Effort | Done when |
|---|---|---|---|
| 1 | Eval tiles (index-stamped) + `evaluate.py` | 1 day | IoU table prints for any labelled LAS |
| 2 | `features.py` (ExG/intensity/returns) | ½ day | memmaps exist; ExG histogram shows veg bump |
| 3 | `ground.py` (CSF → DTM → HAG) | 1 day | DTM image is bare terrain |
| 4 | **Rule-only baseline scored** — the §6.5 veto rules alone as a classifier (five numpy comparisons, no neural net) | ½ day | IoU numbers recorded |
| 5 | `ortho.py` (RGB ortho + grids + tiles, inpainted voids) | ½–1 day | ortho looks like an aerial photo |
| 6 | `/segment_batch` (raw scores, shared embedding) + `segment_sam3.py` | 1–2 days | per-class confidence grids for all tiles |
| 7 | `fuse.py` + Z-aware `map_back.py` (+ confidence in LAS) | 1 day | labelled.las opens in CloudCompare, looks sane |
| 8 | Tune prompts/thresholds/vetoes against eval | ongoing | beats the step-4 baseline |

Steps 1–4 ≈ 3 days and produce a working, **measured** classifier with zero
neural networks. Steps 5–7 are the research contribution, measured against a
real bar — and the gap between the baseline and the SAM3 pipeline is a
result for the write-up either way, whichever direction it points.

As v2 stages land, the v1 reference files they replace are deleted:
`slice.py` (absorbed into ortho.py at step 5), `segment.py` (step 6),
`classify.py` (step 7). `map_back.py` is modified in place.

Housekeeping: `.gitignore` covers `data/`, `models/`, `.aider*`,
`__pycache__` — the 16 GB LAS and checkpoints never enter git.

---

## 9. Running the pipeline (end state)

```bash
# one-time: two environments
python3.11 -m venv .venv && source .venv/bin/activate
pip install laspy numpy opencv-python cloth-simulation-filter matplotlib
# mlx_sam3 has its own env — see segmentation/mlx_sam3/README.md (uv sync)

# start the SAM3 backend (its own terminal / its own env)
cd segmentation/mlx_sam3/app/backend && python main.py

# run everything
python main.py --stage all
# or stage by stage
python main.py --stage features
python main.py --stage ground
python main.py --stage ortho
python main.py --stage segment
python main.py --stage fuse
python main.py --stage map_back
python main.py --stage evaluate

# inspect
open data/output/labelled.las      # CloudCompare
```

Interactive mode (last, only after the CLI pipeline is solid): the SAM3
Studio web app in `segmentation/mlx_sam3/app/` gains a LiDAR mode — load the
ortho instead of an uploaded photo, prompt/box interactively, export the
accepted masks through fuse + map_back. The FastAPI backend and Next.js
frontend already exist; this is two endpoints, not a rebuild.

---

## 10. Enhancement register

Improvements folded into this manual, and the deferred ones, in one place:

**Adopted (already specified above):**
- Eval matching by stamped original index, never file order (§6.0) — the
  highest-consequence fragile spot in the original plan.
- Raw scores out of the SAM3 backend; thresholds applied at fuse time (§6.4).
- Image embedding reused across prompts in `/segment_batch` (§6.4).
- `cv2.inpaint` for void fill instead of box-blur (§6.3).
- Per-point confidence written to LAS `user_data` (§6.6).
- Majority-filter cleanup of `label_grid` (§6.5).
- Veto-rejection-rate logging per class (§6.5).

**Deferred (revisit when the coarse pipeline is measured and stable):**
- Read units/CRS from the LAS header instead of hardcoding US survey feet.
- Guardrail / curb classes (~1 px at 0.5 ft/px; needs finer resolution or a
  different sensor pass).
- Intensity-based vetoes (asphalt vs concrete separation) — add only if the
  confusion matrix shows road/sidewalk bleeding.
- Multi-tile / multi-site generalisation: the chunked design already scales;
  what's missing is only per-site config.
- Parallelising chunk passes — **measured** (`/usr/bin/time -l` on
  `features.py`/`ortho.py` over 100M real points): both are CPU-bound, not
  IO-bound (user time dominates sys time in both). Still not worth
  parallelising — the full `features`/`ortho` stages already run in 8s/35s
  over all 411.5M points, well below any threshold worth the complexity.

---

## Appendix — Why v1 failed (condensed)

1. **Wrong segmentation paradigm:** SAM2 *automatic* mask generation is
   class-agnostic — unlabelled blobs, with all semantics pushed onto a
   downstream classifier that was never trained (no weights existed; it ran
   brittle elevation-rule heuristics). SAM3 text prompts collapse
   segmentation + classification into one grounded step.
2. **Out-of-distribution input:** the false-colour elevation/relief/slope
   composite resembles nothing in a vision model's training data. The RGB
   ortho does — the Temple-Texas result is the proof.
3. **Ignored the richest data:** format-8 LAS carries RGB/intensity/
   returns; v1 read XYZ only.
4. **Column-wise map-back:** every point in a vertical column got the
   surface pixel's label — ground under trees became "vegetation" even when
   the 2D mask was perfect. Fixed by `surface_z` + HAG rules (§6.6).
5. **Unmeasurable:** no ground truth, no metric — "better" and "different"
   were indistinguishable. Fixed by §6.0.

Kept from v1: grid binning + `grid_meta.npz` invertible mapping, the SAT box
filter, chunked LAS writing, priority painting, the `--stage` CLI.
Deleted as v2 lands: SAM2 AMG stage, VGG19 classifier + `train()`, elevation
heuristic, false-colour composite as model input.
