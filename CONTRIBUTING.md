# Contributing / Development Setup

This repo needs **two separate Python environments**. Mixing them (installing everything into one `.venv`) is not supported — `segmentation/mlx_sam3` has its own dependency set (MLX, a separate PyTorch-adjacent stack) managed by `uv`, independent of the root pipeline's `requirements.txt`.

## 1. Root pipeline environment

Runs `main.py` and every stage except `segment` (`features`, `ground`, `ortho`, `fuse`, `map_back`, `baseline`, `evaluate`).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the automated checks:

```bash
pip install pytest ruff   # dev tools, not in requirements.txt
python -m pytest -q tests
ruff format --check classification evaluation projection reprojection segmentation/segment_sam3.py config.py main.py
ruff check classification evaluation projection reprojection segmentation/segment_sam3.py config.py main.py
```

`tests/test_self_checks.py` wires up every pipeline module's own `--self-check`. One check (`features.py`'s) reads the real source LAS file if it's present locally and is skipped automatically when it isn't (e.g. on CI runners) — it never writes to the real `data/derived/` directory either way.

## 2. SAM3 segmentation backend

Only needed for `main.py --stage segment` (or `--stage all`). This is a separate, `uv`-managed project at `segmentation/mlx_sam3/` — see [`segmentation/mlx_sam3/README.md`](segmentation/mlx_sam3/README.md) for its own feature set and details.

```bash
cd segmentation/mlx_sam3
uv run python app/backend/main.py   # starts the FastAPI backend on :8000
```

`segment_sam3.py` in the root pipeline talks to this backend over HTTP at `config.SAM3_URL` (default `http://localhost:8000`) — it does not import anything from `mlx_sam3` directly. If the backend isn't running, `--stage segment` fails immediately with a clear message rather than hanging.

(`segmentation/mlx_sam3/app/run.sh` also starts an interactive web UI on top of the same backend — useful for manually exploring SAM3 prompts, not required for the pipeline itself.)

## Making changes

- Match the existing style: chunked/memmap processing for anything touching the point cloud (never load the full ~411M-point dataset into memory at once), every constant lives in `config.py` (MANUAL.md §4), every pipeline module carries a synthetic `_self_check()` runnable via `--self-check`.
- Run `python -m pytest -q tests` and the `ruff` commands above before opening a PR — CI runs the same checks (`.github/workflows/ci.yml`).
- `MANUAL.md` is the authoritative pipeline spec; `strategy.md` documents the build order and the AI-collaboration conventions used to build this repo. Update `MANUAL.md` alongside any change to a stage's actual behavior.
