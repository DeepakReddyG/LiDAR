# LiDAR Point Cloud Classification Project

## Overview

This project is a research-oriented pipeline for LiDAR point cloud classification using modern computer vision and AI techniques.

The pipeline:
- Loads large-scale LiDAR point cloud datasets
- Converts 3D point cloud data into 2D projections
- Applies AI/computer vision models for object classification
- Maps the classified results back into 3D space

It is intended as a research-oriented prototype for intelligent LiDAR scene understanding in civil engineering and infrastructure applications.

**For the authoritative pipeline specification** (stage-by-stage design, class set, thresholds, data contracts), see [`MANUAL.md`](MANUAL.md). For the build history and AI-collaboration playbook, see [`strategy.md`](strategy.md). This README is a project summary and quick-start guide, not the spec — if this file and `MANUAL.md` ever disagree, `MANUAL.md` is authoritative.

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

The pipeline is a six-stage CLI driven entirely by `config.py`:

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
python main.py --stage features           # or run a single stage
python main.py --stage ground
python main.py --stage ortho
python main.py --stage segment
python main.py --stage fuse
python main.py --stage map_back

python main.py --stage baseline           # rule-only classifier on eval tiles
python main.py --stage evaluate --gt <gt.las> --pred <pred.las>
```

Each stage checks its own prerequisites (per `main.py`'s `STAGES` table) and fails fast with a clear message naming the missing file if you run stages out of order.

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
