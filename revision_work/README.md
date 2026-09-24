# Reproducible bounded revision — September 23, 2026

Start with [`../research_paper_documents_and_drafts/results_summary.md`](../research_paper_documents_and_drafts/results_summary.md). The branch is `revision/2026-09-23`, created from `exp` with inherited working-tree changes preserved. Source snapshots separate inherited code from the corrected run. Historical outputs and failed integration attempts remain in `archive/`.

## Scope and safeguards

`holdout.json` and `objective.md` were committed before model runs. Development uses only the whitelisted tile-C file and the fixed 40 × 30 ft reviewed window. Bounded preprocessing uses 2,505,912 context points and every one of the 158,618 reviewed pilot identities. Original references are read-only and source hashes are checked. None of the experiments runs the full survey.

The final selected configuration and source hashes were frozen in `manifests/final_manifest.json`, commit `d48d085ec9dd1cfd5a55671c2871f0a02b5b3567`, before any held-out data access. `phase5_gate.json` and its completion record consume the final event. **Do not rerun the Phase 5 command or reset its gate.** Both held-out reference files were absent; only blind annotation preparation occurred. Independent annotation and a future evaluation require a newly approved protocol.

## Environment and exact backend

Use the repository `.venv/bin/python` (Python 3.13.15 in this execution). Each manifest records package versions, instantiated CSF settings, every configured threshold/prompt, source hashes, preprocessing, checkpoint hash, seed and cache policy. SAM3 ran through the local MLX/Metal backend, with no HTTP service or downloads. The required checkpoint SHA-256 is `0ad4c3f42ecf706c4cda63cf58d621699491ed65012b3999284ea370984f7173`. Its lexical `.safetensors` path is retained because MLX infers the format from the suffix. Model peak allocation is not whole-process RSS.

Raw inputs and the checkpoint are not redistributed. Their source/provider, acquisition, colorization and CRS facts remain unknown where not supplied. Paths in the existing manifests refer to this machine; portable access and licensing require the provider/PI.

## One-command executions

These are the commands used for clean output directories, from the repository root. Existing outputs deliberately prevent overwriting. For a new *development-only* reproduction, retain these records and supply a separately saved manifest with a new output directory under `revision_work/runs/`; source/data/protocol identities and registered parameters must remain exact. A manifest freezes executable hashes, so run its recorded code version.

```bash
MPLCONFIGDIR=/private/tmp/lidar-revision-mpl .venv/bin/python -m revision.runner all --manifest revision_work/manifests/phase1_fresh_v3.json
MPLCONFIGDIR=/private/tmp/lidar-revision-mpl .venv/bin/python -m revision.experiments run --manifest revision_work/manifests/fixed_anchor.json
MPLCONFIGDIR=/private/tmp/lidar-revision-mpl .venv/bin/python -m revision.experiments matrix --manifest revision_work/manifests/fixed_anchor.json --full
```

The first command executes features, terrain, orthographic image, fresh SAM3, fusion, point transfer/export and strict scoring. The second repeats the corrected pipeline. The third computes the 28 registered component/rule/downstream sensitivity outcomes using only the corrected run's verified identical-image responses; it writes `analysis_full/` separately. Four raster sensitivities use the same `experiments run` command with `grid_025.json`, `grid_1.json`, `rgb_band_075.json` or `rgb_band_3.json`. These regenerate images and SAM3 responses rather than reusing another raster's masks.

A fresh clone needs the nontracked raw inputs, checkpoint and saved run artifacts for report-only regeneration. Binary clouds/arrays are excluded from Git. The complete numeric metric evidence and provenance are tracked in `evidence/`; source scripts, manifests and gate records are tracked. The canonical document folder is also Git-ignored, so keep or deliberately package it when sharing.

## Reports, tests, and artifacts

```bash
MPLCONFIGDIR=/private/tmp/lidar-revision-mpl .venv/bin/python -m pytest tests/test_revision_*.py -q
MPLCONFIGDIR=/private/tmp/lidar-revision-mpl .venv/bin/python -m revision.paper_outputs
.venv/bin/python -m revision.results_summary
.venv/bin/python -m revision.replacement_text --phase5-record revision_work/phase5_gate.json.completion.json --final-manifest revision_work/manifests/final_manifest.json
```

The recorded revision suite has 177 passing tests. It does not claim the entire legacy suite was run; legacy global-data self-checks were deliberately excluded to protect held-out scope. The report generators use saved numbers and archive superseded document outputs. Do not run model experiments or the final gate merely to regenerate figures.

- `evidence/`: full measured metrics, independent fix audit, data quality, source/output hashes and timing records.
- `runs/phase1_fresh/output/`: original fresh predictions and arrays.
- `runs/fixed_anchor/analysis/`: corrected anchor, separate LAS prediction provenance and pilot arrays.
- `runs/fixed_anchor/analysis_full/`: 28 component outputs and per-variant predictions/confusion/strata/routes.
- `runs/{grid_025,grid_1,rgb_band_075,rgb_band_3}/analysis/`: four fresh-raster sensitivities.
- `annotation_protocol.md`: blinded human audit and independent reference instructions. Software creates no semantic references.
- `../research_paper_documents_and_drafts/supporting/revision_experiments_2026-09-23/`: generated tables and separate pilot/held-out annotation packages. Give annotators only `annotator/`, keeping `private/` identity mappings and every model/reference result separate.
- `../research_paper_documents_and_drafts/images_and_charts/revision_experiments_2026-09-23/`: six figure families, captions and all 32 confusion matrices.

The old HTTP filename-only cache is not a verified reproduction path. The revision worker only consumes explicitly hashed images/checkpoints/source identities and refuses stale/incomplete responses. No historical full-survey cache was reused.
