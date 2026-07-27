# Remaining Fixes Plan — LiDAR Pipeline

**Generated:** 2026-07-27  
**Branch:** `exp`  
**Prerequisite:** The prior improvement work (CI self-sabotage removal, pytest wiring, README/CONTRIBUTING, pinned deps, silent-failure hardening, etc.) must already be **committed**. Do not re-do those fixes.

**Method:** One task at a time, in order. Do not batch tasks. Do not refactor unrelated code. Every task has acceptance criteria and a validation command. Update the Progress Tracker when a task starts/completes/blocks.

**Out of scope for AI-only execution (human-gated):**
- Hand-labelling GT tiles in CloudCompare (Task H1)
- Force-push / git history rewrite (Task H2)
- Merging `exp` → `main` / retiring `dev` (Task H3)
- Extracting `mlx_sam3` as a submodule (Task H4)

---

## Progress Tracker

| Task | Title | Priority | Status | Notes |
|---|---|---|---|---|
| H1 | Hand-label eval tiles (CloudCompare) | Critical | Not Started | Human only |
| 1 | Delete v1 leftover artifacts under `data/` | High | Not Started | |
| 2 | Fix `segment_sam3` cache clobber + f-string | Medium | Not Started | |
| 3 | Replace `map_back` negative-index LUT | Medium | Not Started | |
| 4 | Add `--stage evaluate_all` multi-tile scoring | High | Not Started | Depends on H1 for real scores; code can ship first |
| 5 | Bind SAM3 backend to `127.0.0.1` by default | Medium | Not Started | |
| 6 | Make torch/torchvision optional in mlx_sam3 | Medium | Not Started | |
| 7 | Optimize `majority_filter` (keep numpy-only) | Medium | Not Started | |
| 8 | Archive/mark `NOTES.md` as historical | Low | Not Started | |
| 9 | Verify remote CI green after commit/push | High | Not Started | Needs push |
| H2 | Purge large blobs from git history | High | Not Started | Human approval |
| H3 | Resolve `exp` / `main` / `dev` divergence | High | Not Started | Human decision |
| H4 | mlx_sam3 submodule / demo-asset cleanup | Optional | Not Started | Human decision |

Status: `Not Started` / `In Progress` / `Blocked` / `Completed`

---

## Sequential Execution Rules

1. Paste this file’s **Global Context** at the start of each session, then **one task block**.
2. Finish a task only when its Validation Steps pass.
3. Prefer patterns already in the repo (`config.py` constants, `_self_check()`, `--stage` CLI, chunked laspy).
4. No new dependencies unless the task explicitly allows them. Standing order from `strategy.md`: laspy, numpy, matplotlib, pillow, opencv-python-headless, cloth-simulation-filter, requests only — anything else needs human yes.
5. After code changes: `source .venv/bin/activate && python -m pytest -q tests` must stay green (currently 9 tests).
6. Do not commit unless the user asks.

### Global Context

```
/context
- Repo: LiDAR point cloud classification, branch `exp`. Authoritative design: MANUAL.md.
- Pipeline CLI: python main.py --stage features|ground|ortho|segment|fuse|map_back|baseline|evaluate|all
- Config single source of truth: config.py
- Tests: tests/test_self_checks.py (pytest). Run: python -m pytest -q tests
- SAM3 backend: segmentation/mlx_sam3/ (separate env); pipeline calls http://localhost:8000
- Already done (do NOT redo): CI intentional-failure removal, pytest wiring, README rewrite,
  CONTRIBUTING.md, pinned requirements.txt, HTTP error hardening in segment_sam3,
  magic-number consolidation into config.py, silent mask-shape WARNING, features self-check
  using tempfile, MANUAL/strategy status banners.

/constraints
- Never load all 411M points into RAM.
- Do not delete or regenerate data/derived/, data/slices/ ortho grids, data/masks/conf_*.npy,
  or data/output/labelled.las unless a task explicitly says so.
- Task 1 deletes only the listed v1 leftover filenames — nothing else.
```

---

## Task H1 — Hand-label eval tiles (HUMAN ONLY)

**Priority:** Critical  
**Why:** Without `tile_{a,b,c}_gt.las`, no IoU exists. Baseline and SAM3 pipeline are unmeasured. T7 tuning cannot start.

**Steps (human):**
1. Open each of `data/eval/tile_a.las`, `tile_b.las`, `tile_c.las` in CloudCompare.
2. Segment and assign LAS classification codes:
   - `11` = pavement / sidewalk / parking
   - `3` = grass
   - `5` = tree
   - `6` = building
   - `64` = vehicle
   - `1` = unassigned
3. Save as `data/eval/tile_{a,b,c}_gt.las` (preserve `orig_index` extra dimension — CloudCompare may reorder points; that is fine).
4. Then run:
   ```bash
   source .venv/bin/activate
   python main.py --stage baseline
   # after Task 4 lands:
   python main.py --stage evaluate_all
   ```

**Done when:** All three `*_gt.las` exist; `data/eval/baseline_scores.json` exists with non-all-zero IoU for at least pavement and tree.

**AI must not:** invent synthetic GT and claim the pipeline is validated.

---

## Task 1 — Delete v1 leftover artifacts under `data/`

**Priority:** High  
**Category:** Hygiene

**Problem:** v1 outputs still occupy ~1.6 GB and confuse which files are current:
- `data/slices/elevation_grid.npy`
- `data/slices/elevation_grid_raw.npy`
- `data/slices/top_down.png`
- `data/masks/masks.npy` (~732 MB)
- `data/masks/mask_meta.npy`
- `data/masks/class_labels.npy`
- `data/masks/labelled_overlay.png`
- `data/masks/overlay.png`
- `data/output/labelled_points.npz` (~876 MB) — replaced by `labels.npy` + `labelled.las` per `map_back.py` docstring

**Do NOT delete:** `ortho_rgb.png`, `surface_z.npy`, `*_grid.npy`, `void_mask.npy`, `tiles/`, `grid_meta.npz`, `conf_*.npy`, `label_grid.npy`, `conf_grid.npy`, `veto_stats.json`, `labelled.las`, `labels.npy`, anything under `data/derived/` or `data/eval/` or `data/raw/`.

**Required changes:**
1. Delete only the filenames listed above (if present).
2. Update `.gitignore` to ignore those v1 names explicitly so they cannot be re-added accidentally, e.g.:
   ```
   data/slices/elevation_grid.npy
   data/slices/elevation_grid_raw.npy
   data/slices/top_down.png
   data/masks/masks.npy
   data/masks/mask_meta.npy
   data/masks/class_labels.npy
   data/masks/*_overlay.png
   data/masks/overlay.png
   data/output/labelled_points.npz
   ```

**Validation:**
```bash
# all of these must fail (file absent)
test ! -e data/masks/masks.npy
test ! -e data/output/labelled_points.npz
test ! -e data/slices/elevation_grid.npy
# v2 outputs still present
test -e data/masks/label_grid.npy
test -e data/output/labelled.las || test -e data/output/labels.npy
python -m pytest -q tests
```

**Acceptance:** ~1.6 GB reclaimed; `data/` maps 1:1 to v2 stage outputs; suite green.

---

## Task 2 — Fix `segment_sam3` cache clobber + cosmetic f-string

**Priority:** Medium  
**Files:** `segmentation/segment_sam3.py`

**Problem A — cache clobber:** When a cache JSON exists but is missing some prompts, the code re-runs inference and then `cache.write_text(json.dumps(data))`, which **overwrites** previously cached prompts. Adding a class later wipes the other classes’ cached detections.

**Fix:** When merging a fresh response into an existing cache, union `results` dicts:
```python
# pseudocode — adapt to existing variable names
if cache.exists():
    old = json.loads(cache.read_text())
else:
    old = {"results": {}}
# after successful POST:
merged = {**old, **data}  # keep top-level keys from new response
merged["results"] = {**old.get("results", {}), **data["results"]}
cache.write_text(json.dumps(merged))
return merged
```
Only re-request prompts that are actually missing (optional optimization; full re-request + merge is also fine).

**Problem B — f-string:** Line printing `coverage>{0.1}` — replace with literal text:
```python
print(f"  {out.name}: coverage>0.1 {(g > 0.1).mean():.1%}  max {g.max():.2f}")
```

**Validation:**
1. Extend `_self_check()` or add a tiny unit test that:
   - Writes a fake cache with prompt `"road"` only.
   - Simulates a merge of new results for `"tree"`.
   - Asserts both `"road"` and `"tree"` remain in `results`.
2. `python -m pytest -q tests`
3. Confirm print format no longer uses `{0.1}` braces (grep).

**Acceptance:** Adding prompts cannot wipe old cache entries; log line is readable.

---

## Task 3 — Replace `map_back` negative-index LUT

**Priority:** Medium  
**Files:** `reprojection/map_back.py`

**Problem:** `code_lut = np.full(max_id + 2, …)` then `code_lut[labels]` relies on `labels == -1` wrapping to the last slot. Fragile if class ids change.

**Fix:** Map explicitly without wrap-around magic:
```python
# Build lut only for non-negative ids
max_id = max(_NAME_TO_ID.values())
code_lut = np.full(max_id + 1, UNLABELLED_LAS_CODE, np.uint8)
for cid, code in id_to_code.items():
    if cid >= 0:
        code_lut[cid] = code

# When writing:
las_codes = np.full(n, UNLABELLED_LAS_CODE, np.uint8)
valid = labels >= 0
las_codes[valid] = code_lut[labels[valid]]
ch.classification = las_codes
```
Update the class-distribution `counts` path similarly so `-1` is counted without relying on wrap.

**Validation:**
1. `python -m reprojection.map_back --self-check` (or via pytest).
2. Add/extend self-check: a batch with labels `[-1, tree_id]` produces `UNLABELLED_LAS_CODE` and tree’s LAS code respectively.
3. Full `pytest -q tests`.

**Acceptance:** No negative-index wrap; unlabelled points always get `UNLABELLED_LAS_CODE`.

---

## Task 4 — Add `--stage evaluate_all`

**Priority:** High  
**Files:** `main.py`, possibly `evaluation/evaluate.py`, `config.py` if needed

**Problem:** Scoring all three GT tiles requires three manual `evaluate --gt … --pred …` invocations. Tuning loop is slowed.

**Required behavior:**
```bash
python main.py --stage evaluate_all
```

Behavior:
1. For each name in `EVAL_TILE_BOUNDS` (`a`, `b`, `c`):
   - GT path: `EVAL_DIR / f"tile_{name}_gt.las"`
   - Pred paths to score (if present):
     - baseline: `EVAL_DIR / f"tile_{name}_baseline.las"`
     - optionally also score full-site output filtered by orig_index **only if** a simple path already exists — do **not** invent a new map-back of eval tiles unless trivial. Minimum viable: score baseline preds; if you can score `labelled.las` via orig_index match without loading all 411M points into RAM, do so — otherwise document that SAM3 scoring against GT requires cropping labelled points by `orig_index` from `labels.npy` (memmap) into a temp LAS, or scoring baseline only until a follow-up.
2. **Pragmatic MVP (preferred):**  
   - Score each existing `tile_{name}_baseline.las` against `tile_{name}_gt.las`.  
   - Also, if `OUTPUT_DIR / "labels.npy"` exists, build a small predicted LAS for each eval tile by reading `tile_{name}.las`’s `orig_index`, looking up labels in the memmap, writing classification codes, and evaluating — this avoids reading the 16GB `labelled.las`.
3. Write combined JSON:
   - `data/eval/baseline_scores.json`
   - `data/eval/sam3_scores.json` (if labels.npy path implemented)
4. If a GT file is missing, print a clear message naming Task H1 / CloudCompare and **continue** other tiles (or exit non-zero only if zero GTs found).

Wire into `main.py` `STAGES` / argparse choices. Prereq check: at least `EVAL_DIR` exists.

**Validation:**
```bash
python main.py --help   # must list evaluate_all
# Without GT: must print clear "missing GT" guidance, not a traceback
python main.py --stage evaluate_all
python -m pytest -q tests
```
With GT present (after H1): JSON files written; IoU table printed.

**Acceptance:** One command scores all available GT tiles; missing GT is a message, not a crash.

---

## Task 5 — Bind SAM3 backend to loopback by default

**Priority:** Medium  
**Files:** `segmentation/mlx_sam3/app/backend/main.py` (and README note if it claims `0.0.0.0`)

**Problem:** `uvicorn.run(app, host="0.0.0.0", port=8000)` exposes the model API on all interfaces with no auth.

**Fix:**
```python
import os
host = os.environ.get("SAM3_HOST", "127.0.0.1")
port = int(os.environ.get("SAM3_PORT", "8000"))
uvicorn.run(app, host=host, port=port)
```
Optionally reject `/segment_batch` bodies over a max size (e.g. 20 MB decoded image) — nice-to-have, not required if timeboxed.

**Validation:** Grep shows default host is `127.0.0.1`. Document in `segmentation/mlx_sam3/app/README.md` or main mlx README: set `SAM3_HOST=0.0.0.0` only when intentional.

**Acceptance:** Default bind is loopback; opt-in for LAN via env var.

---

## Task 6 — Make torch/torchvision optional in mlx_sam3

**Priority:** Medium  
**Files:** `segmentation/mlx_sam3/pyproject.toml`

**Problem:** `torch` and `torchvision` are listed as hard dependencies for an MLX-native runtime — multi-GB install bloat.

**Fix:** Move them to an optional extra:
```toml
[project.optional-dependencies]
convert = ["torch>=2.9.1", "torchvision>=0.24.1"]
```
Remove them from the main `dependencies` list **only after** verifying:
```bash
cd segmentation/mlx_sam3
# in that env, without torch:
python -c "from sam3 import build_sam3_image_model; print('ok')"
```
If import fails without torch, **stop** and mark Blocked — do not break the backend. Report what imports torch.

**Validation:** `pyproject.toml` has torch only under optional-dependencies; core import works OR task marked Blocked with evidence.

---

## Task 7 — Optimize `majority_filter` (numpy-only)

**Priority:** Medium  
**Files:** `classification/fuse.py`

**Problem:** Triple loop over `size × size × n_values` is slow on ~1700×2100 grids. Fuse is re-run during tuning.

**Constraints:** Do **not** add scipy. Stay numpy-only.

**Approach (pick one, simplest that passes tests):**
1. Keep the count-tensor idea but remove the Python loop over `values` by using a denser encoding: shift labels to `0..K-1`, then for each window offset accumulate into `counts[label, :, :]` via advanced indexing — still O(size²) passes but no Python over unique values.
2. Or: only run majority filter on pixels that differ from a neighbor (speckle candidates) — more complex; prefer (1).

**Validation:**
1. Existing `test_fuse_self_check` and `test_fuse_priority_order_isolated` pass.
2. Optional: time old vs new on a synthetic 2000×2000 grid — new should be clearly faster (print both).
3. Output of fuse on current `data/masks/conf_*.npy` (if present) should match previous `label_grid` for an identical config — if you can load pre-change grid from git, byte-compare; else rely on self-checks.

**Acceptance:** Same semantics; faster implementation; no new deps; tests green.

---

## Task 8 — Archive/mark `NOTES.md` as historical

**Priority:** Low  
**Files:** `NOTES.md`, optionally `README.md` one-line pointer

**Problem:** `NOTES.md` describes v1-era tools (SAM2, open3d, VGG19, etc.) and contradicts `MANUAL.md`.

**Fix:** Add a banner at the top:
```markdown
> **Historical notes (pre-v2).** Superseded by [`MANUAL.md`](MANUAL.md).
> Do not treat this file as the current pipeline spec.
```
Do not delete content unless the user asks.

**Validation:** Banner present; `MANUAL.md` still authoritative per README.

---

## Task 9 — Verify remote CI green after commit/push

**Priority:** High  
**Prerequisite:** User has committed and pushed the prior improvement commit (and any of Tasks 1–8 they want included).

**Steps:**
1. Confirm workflows no longer contain intentional failure (already done in tree).
2. After push: `gh run list --branch exp --limit 10` and confirm `ci.yml` / `security.yml` / etc. succeed (or `workflow_dispatch`).
3. If CI fails on ruff: fix only the failures introduced by remaining-fix tasks.
4. Do **not** enable branch protection unless the user asks (needs admin).

**Acceptance:** At least `Python CI` workflow completes successfully on `exp`.

---

## Task H2 — Purge large blobs from git history (HUMAN GATE)

**Do not run** `git filter-repo` / BFG / force-push unless the user explicitly approves in writing in the chat.

If approved later: inventory large blobs with `git rev-list --objects --all | git cat-file --batch-check=…`, remove LAS/weights from history, force-push with coordination. Until then: **Blocked**.

---

## Task H3 — Resolve branch divergence (HUMAN DECISION)

Facts:
- `main` and `dev` are at `e9ac1c01` (pre-v2).
- All v2 work lives on `exp` only.

Options for the human:
A. Fast-forward / merge `exp` → `main` and delete or freeze `dev`.  
B. Keep `exp` as long-lived integration branch; protect it.  
C. Open a PR `exp` → `main` and review.

AI must not merge or delete branches without explicit instruction.

---

## Task H4 — mlx_sam3 submodule / demo assets (HUMAN DECISION)

Deferred: which `IMG_*.jpg/png` and notebooks to keep; whether to extract `segmentation/mlx_sam3` as a git submodule. No speculative deletion.

---

## Suggested AI session order

1. Task 1 (v1 file delete) — safe, high clarity  
2. Task 2 (cache + f-string)  
3. Task 3 (map_back LUT)  
4. Task 4 (`evaluate_all`)  
5. Task 5 (bind address)  
6. Task 6 (torch optional) — may Block  
7. Task 7 (majority_filter)  
8. Task 8 (NOTES banner)  
9. Task 9 (CI verify) — after user pushes  
10. Remind user about H1 (GT labelling) — highest research value  

---

## Definition of done (this plan)

- [ ] Tasks 1–8 Completed or explicitly Blocked with evidence  
- [ ] `python -m pytest -q tests` green  
- [ ] No v1 leftover files listed in Task 1  
- [ ] `python main.py --help` lists `evaluate_all`  
- [ ] SAM3 backend defaults to `127.0.0.1`  
- [ ] User informed that **H1 (GT labelling)** remains the critical research blocker  
- [ ] User informed that H2/H3/H4 need human decisions  
