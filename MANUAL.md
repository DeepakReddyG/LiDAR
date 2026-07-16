# Implementation Manual — `exp` branch

The proper pipeline: LAS point cloud → true-colour nadir ortho → **SAM3
text-prompted segmentation** (your `mlx_sam3` port) → physics veto with
LiDAR-derived features → Z-aware map-back → labelled LAS, scored against
hand-labelled tiles.

This replaces the v1 design (SAM2 automatic masks + untrained classifier).
The v1 diagnosis is summarised at the end; the short version is that v1 used
class-agnostic masks on an out-of-distribution false-colour image, classified
by untrained heuristics, and mapped labels down entire vertical columns.

---

## Part 0 — What your SAM3 experiments already established

Your `segmentation-with-sam` results are the design evidence. Bake these in
as rules, not preferences:

| Finding | Evidence | Design rule |
|---|---|---|
| Nadir ortho works; oblique fails | oblique @0.5: "road"=2 obj incl. whole-image box; nadir @0.3: clean road network, 13 parking lots | **Segment only the top-down orthographic raster.** Never oblique renders. |
| Amorphous surfaces need low thresholds; objects don't | road/parking useless @0.5, good @0.3; cars/trees/buildings fine @0.5 | **Per-class thresholds**, not one global value. |
| Whole-image boxes are a known failure mode | oblique "road" 0.60 box spanning the frame | Drop detections whose box covers > 80 % of the tile. |
| SAM3 handles colourised point-cloud rasters | Temple-Texas ortho segmented well despite scan-line voids | RGB ortho from the LAS is a valid model input. **Fill the black voids first** — they're the one artefact a photo-trained model never saw. |
| Text prompts return *labelled* masks | every panel | Segmentation and classification are **one stage**. No per-mask classifier exists in this pipeline. |

---

## Part 1 — Dataset facts that shape the design

`UPark_Merged_PS_NAD83_G18_USFT_las.las` — **LAS point format 8**, 411.5 M
points, ~843 × 1058 ft site, ~460 pts/ft², all currently unclassified. Every
point carries:

- **RGB (16-bit)** → the true-colour ortho for SAM3
- **NIR** → per-point **NDVI = (NIR−R)/(NIR+R)** — vegetation nearly free
- **Intensity** (17k–61k) → asphalt vs concrete/grass separation
- **Return number / count (1–5)** → multi-return = canopy
- Units are **US survey feet** — all distance thresholds below are in ft

v1 read only XYZ. The spectral attributes are what make the veto stage and
the under-canopy labelling possible.

---

## Part 2 — Architecture

```
                     LiDAR repo (this repo, python 3.11 / torch env)
 ┌──────────────────────────────────────────────────────────────────────┐
 │ features.py   chunked LAS pass → ndvi / intensity / nreturns (memmap)│
 │ ground.py     CSF ground filter → DTM grid → per-point HAG           │
 │ ortho.py      RGB nadir ortho (+ void fill) + surface_z / ndvi /     │
 │               hag / intensity grids + grid_meta + 1024px tiles       │
 └───────────────┬──────────────────────────────────────────────────────┘
                 │ tiles (PNG) + prompts + per-class thresholds
                 ▼            HTTP (or subprocess) — process boundary,
 ┌───────────────────────────┐ because mlx_sam3 lives in its own
 │ mlx_sam3 (your repo,      │ python 3.14 / MLX environment
 │ FastAPI backend)          │
 │ POST /segment_batch       │ → per-tile, per-prompt masks + scores
 └───────────────┬───────────┘
                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ segment_sam3.py  stitch tiles → per-class confidence grids           │
 │ fuse.py          physics veto (NDVI/HAG/returns) + priority painting │
 │                  → label_grid.npy                                    │
 │ map_back.py      Z-aware, chunked → labelled.las                     │
 │ evaluate.py      per-class IoU vs hand-labelled tiles                │
 └──────────────────────────────────────────────────────────────────────┘
```

**Class set** (coarse first, per the professor's guidance; fits a campus/park
site):

| id | class | prompt | threshold | LAS code |
|---|---|---|---|---|
| 0 | pavement/road | `"road"` | 0.30 | 11 (road surface) |
| 1 | sidewalk/path | `"sidewalk"` | 0.30 | 11 |
| 2 | parking lot | `"parking lot"` | 0.30 | 11 |
| 3 | grass | `"grass"` | 0.35 | 3 (low veg) |
| 4 | tree | `"tree"` | 0.45 | 5 (high veg) |
| 5 | building | `"building"` | 0.50 | 6 (building) |
| 6 | vehicle | `"car"` | 0.50 | 64 (custom) |
| −1 | unlabelled | — | — | 1 (unassigned) |

Thresholds start from your threshold=0.3 vs 0.5 experiments; tune only
against `evaluate.py` numbers (Part 4). Guardrail/curb: postponed — ~1 px at
this resolution; revisit only after coarse IoU is good.

---

## Part 3 — Stage-by-stage implementation

### 3.1 `features.py` — per-point spectral features (½ day)

One chunked pass, no dependencies beyond laspy/numpy:

```python
with laspy.open(LAS_PATH) as f:
    ndvi  = np.lib.format.open_memmap("data/derived/ndvi.npy", mode="w+",
                                      dtype=np.float32, shape=(f.header.point_count,))
    # same for intensity.npy (float32), nreturns.npy (uint8)
    off = 0
    for ch in f.chunk_iterator(10_000_000):
        r, n = ch.red.astype(np.float32), ch.nir.astype(np.float32)
        ndvi[off:off+len(ch)] = (n - r) / (n + r + 1e-6)
        ...
        off += len(ch)
```

Arrays are index-aligned with LAS point order — that alignment is the
contract every later stage relies on. ~1.6 GB per float32 array on disk;
memory-mapped, so RAM stays flat.

**Check before proceeding:** plot the NDVI histogram. It should be bimodal
(vegetation vs hard surface). If it isn't, the NIR channel is unreliable and
the veto rules in 3.5 need recalibrating on the eval tiles.

### 3.2 `ground.py` — DTM + Height Above Ground (1 day)

`pip install cloth-simulation-filter` (CSF). 411 M points won't fit CSF —
they don't need to:

1. Decimate: lowest-Z point per 2×2 ft cell (a chunked pass reusing the
   binning pattern from v1 `slice.py`) → ~200k candidate points.
2. CSF on the decimated set (`cloth_resolution ≈ 2.0` ft, start there) →
   ground points.
3. Rasterise ground points to a **DTM grid** (min-Z per cell, gap-fill with
   the v1 box filter — keep `_uniform_filter_2d`, it's good).
4. Per-point `hag.npy` (memmap): `z − dtm[row, col]` in one chunked pass.

**Check:** render the DTM as an image. Smooth terrain only — any embossed
building or tree outline means `cloth_resolution` is off.

### 3.3 `ortho.py` — the raster SAM3 actually sees (½–1 day)

Keep v1's binning and `grid_meta.npz` exactly (that code is correct — it's
the invertible forward/backward mapping the professor flagged as the core
problem, and v1 solved it). Change what fills the cells:

- **`ortho_rgb.png`** — per cell, mean R,G,B of points within 1.5 ft of the
  cell's max Z (top-surface colour, not colour smeared through the canopy).
  16-bit → 8-bit: divide by 256.
- **`surface_z.npy`** — max Z per cell (this drives the Z-aware map-back).
- **`ndvi_grid.npy`, `hag_grid.npy`, `intensity_grid.npy`** — mean per cell
  (for the veto stage).
- **Void fill** — cells with zero points (the black scan lines in your
  Temple-Texas render): fill RGB from the box-filtered neighbourhood mean.
  Also save `void_mask.npy` — veto any label on void pixels at fuse time.
- **Tiles** — 1024×1024 px crops, stride 768 (25 % overlap), saved with their
  grid offsets. At 0.5 ft/px the site is ~1700×2100 px → ~6 tiles. Tiling
  matters because a car is ~30×15 px; downscaling the whole ortho into
  SAM3's input resolution would shrink it below detectability.

Resolution stays 0.5 ft/px.

### 3.4 `segment_sam3.py` — the process boundary (1 day)

Your `mlx_sam3` runs python 3.14 + MLX; this repo runs 3.11 + torch. Don't
merge the environments — call across a boundary. You already have a FastAPI
backend (`app/backend/main.py`); add one headless endpoint to it:

```
POST /segment_batch
  { image: <png base64>, prompts: ["road", "tree", ...],
    thresholds: {"road": 0.30, ...} }
→ { "road": [ {mask_rle, score, bbox}, ... ], "tree": [...], ... }
```

(If HTTP annoys you, a CLI in the sam3 repo — tile dir in, `.npz` of masks
out, driven via `subprocess` — is equally fine. Same contract, pick one.)

On this side, per tile and per class:

1. Drop detections with bbox > 80 % of the tile (your observed failure mode).
2. Union all surviving instance masks into a **semantic confidence grid**:
   per pixel, max score of any covering mask. Instance identity is noise at
   this stage — coarse civil classes need per-pixel semantics, and semantic
   union makes tile stitching trivial (max over overlaps) instead of an
   instance-matching problem.
3. Write one `(H, W) float32` confidence grid per class.

### 3.5 `fuse.py` — physics veto + priority painting (½ day)

SAM3 says what things *look like*; the LiDAR knows what they *are*. Veto
each class grid with the 3.3 stat grids (thresholds are starting points —
calibrate on eval tiles):

| class | keep a pixel only if |
|---|---|
| tree | NDVI > 0.2 **or** HAG > 6 |
| grass | NDVI > 0.15 **and** HAG < 2 |
| building | HAG > 8 |
| road / sidewalk / parking | HAG < 1.5 |
| vehicle | 1 < HAG < 9 |
| any | not on `void_mask` |

Then paint `label_grid.npy` most-specific-first, first claim wins (keeps
v1's one good idea): **vehicle → tree → building → sidewalk → parking →
road → grass**. Log the veto-rejection rate per class — it's a free
diagnostic of whether SAM3 or the thresholds are the weak link.

### 3.6 `map_back.py` — Z-aware, chunked (1 day)

Keep v1's grid indexing and LAS writing; fix the column bug. Per 10 M-point
chunk:

```python
row, col   = <v1 indexing from grid_meta>
surf       = surface_z[row, col]
on_surface = np.abs(z - surf) < 3.0                    # ft
labels[on_surface] = label_grid[row, col][on_surface]

# below-surface points (under canopy, under eaves): LiDAR rules, not SAM3
below = ~on_surface
labels[below & (hag > 2)]                    = TREE     # trunk / understory
labels[below & (hag <= 2) & (ndvi > 0.15)]   = GRASS
labels[below & (hag <= 2) & (ndvi <= 0.15)]  = ROAD_OR_PAVEMENT
```

This is the payoff of the whole design: under a tree, canopy = tree, trunk =
tree, ground = grass or pavement. v1 structurally could not produce that —
it labelled the entire column "vegetation".

Write the class → LAS-code mapping from the Part 2 table into
`classification`, chunked, as v1 already does.

### 3.7 `main.py` — stage runner

Keep v1's `--stage` CLI shape. Stages: `features`, `ground`, `ortho`,
`segment`, `fuse`, `map_back`, `evaluate`. One `config.py` holds every path,
threshold, and prompt — no constants scattered across modules (v1 duplicated
paths in four files).

---

## Part 4 — Evaluation (build FIRST — it is step 1, not step 8)

"Not effective" must become a number before any threshold gets tuned.

1. Crop **3 tiles** (~150×150 ft, laspy XY filter): (a) road/parking,
   (b) buildings, (c) trees over ground. A few M points each.
2. Hand-label in CloudCompare (segment tool → classification code), 2–3 h
   total → `data/eval/tile_{a,b,c}_gt.las`.
3. `evaluate.py` (~60 lines): match points by XY order (crop preserves it),
   print per-class IoU + confusion matrix.

Realistic bars on this data: pavement & tree IoU > 0.8, building > 0.7,
grass > 0.6, vehicle is the hardest. Every prompt/threshold/veto change is
judged by this script and nothing else.

**Optional but recommended baseline:** before wiring SAM3, run the veto
rules of 3.5 *alone* as a classifier (5 numpy comparisons on NDVI/HAG/
returns) and score it. That's the no-neural-net baseline; the SAM3 pipeline
must beat it to justify itself, and the gap between them is a result you can
put in the research write-up either way.

---

## Part 5 — Build order

| # | Task | Effort | Done when |
|---|---|---|---|
| 1 | Eval tiles + `evaluate.py` | 1 day | IoU table prints for any labelled LAS |
| 2 | `features.py` (NDVI/intensity/returns) | ½ day | memmaps exist; NDVI histogram bimodal |
| 3 | `ground.py` (CSF → DTM → HAG) | 1 day | DTM image is bare terrain |
| 4 | Rule-only baseline scored | ½ day | IoU numbers recorded |
| 5 | `ortho.py` (RGB ortho + grids + tiles) | ½–1 day | ortho looks like an aerial photo, voids filled |
| 6 | `/segment_batch` in mlx_sam3 + `segment_sam3.py` | 1–2 days | per-class confidence grids for all tiles |
| 7 | `fuse.py` + Z-aware `map_back.py` | 1 day | labelled.las opens in CloudCompare, looks sane |
| 8 | Tune prompts/thresholds/vetoes against eval | ongoing | beats the step-4 baseline |

Steps 1–4 ≈ 3 days and produce a working, measured classifier with zero
neural networks. Steps 5–7 are the research contribution, measured against a
real bar.

Housekeeping before the first `exp` commit: `.gitignore` for `data/`,
`models/`, `.aider*`, `__pycache__`; don't let the 16 GB LAS or checkpoints
into git.

---

## Appendix — Why v1 failed (condensed)

1. **Wrong segmentation paradigm**: SAM2 *automatic* mask generation is
   class-agnostic — unlabelled blobs, with all semantics pushed onto a
   downstream classifier that was never trained (no weights existed; it ran
   brittle elevation-rule heuristics). SAM3 text prompts collapse
   segmentation+classification into one grounded step.
2. **Out-of-distribution input**: the false-colour elevation/relief/slope
   composite resembles nothing in a vision model's training data. The RGB
   ortho does — your Temple-Texas result is the proof.
3. **Ignored the richest data**: format-8 LAS carries RGB/NIR/intensity/
   returns; v1 read XYZ only.
4. **Column-wise map-back**: every point in a vertical column got the
   surface pixel's label — ground under trees became "vegetation" even when
   the 2D mask was perfect. Fixed by `surface_z` + HAG rules (3.6).
5. **Unmeasurable**: no ground truth, no metric — "better" and "different"
   were indistinguishable. Fixed by Part 4.

Kept from v1: grid binning + `grid_meta.npz` invertible mapping, the SAT box
filter, chunked LAS writing, smallest-first painting, the `--stage` CLI.
Deleted: SAM2 AMG stage, VGG19 classifier + `train()`, elevation heuristic,
false-colour composite as model input.
