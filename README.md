# LiDAR Point Cloud Classification Project

## Overview

This project is a research-oriented pipeline for LiDAR point cloud classification using modern computer vision and AI techniques.

The pipeline:
- Loads large-scale LiDAR point cloud datasets
- Converts 3D point cloud data into 2D projections
- Applies AI/computer vision models for object classification
- Maps the classified results back into 3D space

It is intended as a research-oriented prototype for intelligent LiDAR scene understanding in civil engineering and infrastructure applications.

**For the authoritative pipeline specification** (stage-by-stage design, class set, thresholds, data contracts), see [`MANUAL.md`](MANUAL.md). This README is a project summary and quick-start guide, not the spec — if this file and `MANUAL.md` ever disagree, `MANUAL.md` is authoritative.

---

## For Reviewers

- **Pipeline code:** `main.py` and the stage packages (`projection/`, `segmentation/`, `classification/`, `reprojection/`, `evaluation/`). All settings live in `config.py`. The design is in [`MANUAL.md`](MANUAL.md).
- **Experiments behind the paper:** `revision/` holds the experiment code. [`revision_work/`](revision_work/README.md) holds the run manifests, metric evidence and audit records. `holdout.json` and `objective.md` were committed before any model run, as the preregistered held-out regions and objective.
- **Current evidence:** this is a development pilot on one 40 × 30 ft region of tile C (158,618 reviewed points, tree and grass). The SAM3 pipeline made 5,215 errors and the rule-only baseline made 5,917. No independent held-out evaluation has been run yet. See [`CHANGELOG.md`](CHANGELOG.md) and `revision_work/evidence/`.
- **Data:** the raw LAS survey (~16 GB) and the SAM3 checkpoint are not in this repository.
- **Tests:** `python -m pytest -q tests` runs on synthetic data and needs no survey files.

---

## Research Motivation

LiDAR (Light Detection and Ranging) generates extremely dense 3D spatial datasets representing real-world environments.

Typical LiDAR scans may contain:
- Roads
- Buildings
- Pavements
- Terrain
- Vegetation
- Infrastructure objects

Raw point cloud data is difficult to process directly due to massive data size, unstructured geometry, and complex spatial relationships. Rather than directly classifying 3D geometry, this project projects the point cloud into 2D, applies mature 2D computer vision models, and transfers the results back into 3D space:

```
3D Point Cloud → 2D Projection → AI-Based Image Classification → Projection Back Into 3D
```

---

## The Pipeline (Shipped)

The pipeline is a six-stage CLI driven entirely by `config.py`, preceded by
`grid` setup: derive the shared pixel/world mapping from the LAS header at
`GRID_RESOLUTION` (default 0.5 US survey feet per pixel). Matching grid metadata
is preserved; conflicting metadata stops the run before processing.

1. **`features`** — per-point spectral features (ExG vegetation index, intensity, returns)
2. **`ground`** — CSF ground filter → DTM → per-point Height Above Ground
3. **`ortho`** — RGB nadir orthographic image + statistic grids + tiling
4. **`segment`** — SAM3 text-prompted segmentation → per-class confidence grids
5. **`fuse`** — threshold + physics-veto + priority painting → label grid
6. **`map_back`** — Z-aware relabeling of the original 3D point cloud

Plus a rule-only `baseline` classifier and an `evaluate` stage (per-class IoU + confusion matrix) used to measure the pipeline against hand-labelled ground truth. Full detail on each stage — including the segmentation model choice and why an earlier (SAM2 + elevation-heuristic) approach was replaced — is in [`MANUAL.md`](MANUAL.md).

---

## Dataset Information

Current dataset:

```
data/raw/UPark_Merged_PS_NAD83_G18_USFT_las.las
```

- Size: ~16 GB
- Total points: ~411 million

This is a research-scale LiDAR dataset and requires chunk-based processing instead of full-memory loading. Every stage of the pipeline processes it in fixed-size chunks (`CHUNK_SIZE` in `config.py`) rather than loading it whole.

---

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This covers every stage except `segment`, which needs a second, separately-managed environment for the SAM3 backend — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full two-environment setup and how to run the test suite.

---

## Running the Pipeline

```bash
# start the SAM3 backend first, in its own environment (only needed for --stage segment/all)
cd segmentation/mlx_sam3 && uv run python app/backend/main.py

# then, from the repo root, in the main .venv:
python main.py --stage all                # run the full 6-stage pipeline
python main.py --stage grid               # initialize/validate the shared grid
python main.py --stage features           # or run a single stage
python main.py --stage ground
python main.py --stage ortho
python main.py --stage segment
python main.py --stage fuse
python main.py --stage map_back

python main.py --stage baseline           # rule-only classifier on eval tiles
python main.py --stage evaluate --gt <gt.las> --pred <pred.las>
python main.py --stage evaluate_all       # score available GT tiles vs both methods
```

Each stage checks its own prerequisites (per `main.py`'s `STAGES` table) and fails fast with a clear message naming the missing file if you run stages out of order.

`all` runs grid setup first, so no metadata from an earlier run is required.
When running stages individually, run `grid` before `ground` and `ortho`.

Evaluation supports partial annotation: matched ground-truth points with LAS
codes 0 or 1 are excluded from both IoU and the confusion matrix. Unlabelled
predictions on annotated ground truth still count as errors. The evaluator
reports matched, scored, and ignored point counts, and refuses to score a tile
with no annotated matches. Scores describe only the annotated portion; they do
not establish accuracy on the rest of the tile or survey.

For CloudCompare annotation, first create a safe copy with
`python -m evaluation.annotation prepare --reference data/eval/tile_c.las --output docs/annotation_pilot/prepared/tile_c_annotation.las`.
The original tile stays authoritative. CloudCompare rounds large `orig_index`
values in this dataset; keep the prepared copy's small `tile_index` field during
export, then run `evaluation.annotation restore` before `evaluation.merge_gt_parts`.
See [MANUAL §6.0](MANUAL.md#60-evaluationevaluatepy--build-first-1-day) for the
complete workflow. Preparation and restoration refuse to overwrite files.

---

## Important Notes

### Large Dataset Warning

This dataset contains hundreds of millions of points. Loading the entire dataset simultaneously may exhaust RAM, freeze the system, or cause severe slowdown. Every stage uses chunked/memmap processing instead — do not add code paths that load the full point cloud into memory at once.

---

## Long-Term Research Direction

This project has applications in:
- Civil engineering
- Infrastructure analysis
- Autonomous systems
- Robotics
- GIS
- Smart cities
- Pavement analysis
- Road extraction
- Scene understanding

The long-term objective is to develop scalable AI-assisted LiDAR analysis workflows. See `MANUAL.md` §10 ("Enhancement register") for specific deferred work items (multi-site generalization, guardrail/curb classes, intensity-based vetoes, and others).

---

## Author Notes

This repository is a research-phase project. The core pipeline described above is built and runnable; ongoing work focuses on evaluating and tuning it against hand-labelled ground truth (`MANUAL.md` §6.0, §7) rather than further architectural changes.
