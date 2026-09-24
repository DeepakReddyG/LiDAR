# Revision commits grouped by reviewer item

Branch: `revision/2026-09-23` from `exp`. Scientific/implementation history through `95cffc4e9285bb1575b39d5440977cfdf8d4133f`; the final document-packaging/index commit may follow this generated index. Multi-item commits appear in every applicable group. Unrelated inherited changes were preserved.

## M1 — Prospective reproducibility and source/output identity

- `b4e6313` — M1: preserve inherited pipeline source and working-tree provenance
- `6c3eb61` — M1 M2 M3 M4: add pinned inference, strict metrics and protected evaluation protocol
- `44ee9eb` — M1: make bounded pilot reruns fail closed on provenance and scoring drift
- `1c610d5` — M1: freeze fresh pilot manifest before any inference
- `7bbfae3` — M1: preserve checkpoint format suffix while validating symlink target
- `5a95f7f` — M1: lock corrected checkpoint path before fresh pilot retry
- `84f877f` — M1: retain LAS coordinate scaling in scored pilot subsets
- `0dbae8f` — M1: lock coordinate-safe fresh pilot manifest
- `d1aa127` — M1: record fresh pilot truth and preserve historical comparison
- `bf9f7e4` — M1 M3 M5 M6: fix terrain minimum and preserve export provenance; prepare blind audit
- `0620ab9` — M1 M4 M7: preregister corrected anchor and all raster sensitivities
- `74f59fe` — M1 M5 M6 M7: verify zero metric delta and repaired point provenance

## M2 — Protected spatial regions and one-shot stopping rule

- `5ce1149` — M2 M4: preregister spatial holdouts before revision experiments
- `6c3eb61` — M1 M2 M3 M4: add pinned inference, strict metrics and protected evaluation protocol
- `15c469a` — M2 M3 M4: freeze one-shot annotation branch and deterministic selection
- `d48d085` — M2 M4 M6: freeze final pilot-selected method before opening held-out regions
- `95cffc4` — M2 M3: record single held-out access and stop without missing-reference accuracy

## M3 — Prediction-hidden human annotation preparation

- `6c3eb61` — M1 M2 M3 M4: add pinned inference, strict metrics and protected evaluation protocol
- `bf9f7e4` — M1 M3 M5 M6: fix terrain minimum and preserve export provenance; prepare blind audit
- `2449ccc` — M3 M4 M7: retain all component outcomes and select only by locked objective
- `15c469a` — M2 M3 M4: freeze one-shot annotation branch and deterministic selection
- `95cffc4` — M2 M3: record single held-out access and stop without missing-reference accuracy

## M4 — Matched source/component tests and selection

- `5ce1149` — M2 M4: preregister spatial holdouts before revision experiments
- `2b0cfaf` — M4 M6: preregister balanced error and equal abstention cost
- `6c3eb61` — M1 M2 M3 M4: add pinned inference, strict metrics and protected evaluation protocol
- `208ece4` — M4 M7: freeze matched components and bounded sensitivity execution
- `0620ab9` — M1 M4 M7: preregister corrected anchor and all raster sensitivities
- `2449ccc` — M3 M4 M7: retain all component outcomes and select only by locked objective
- `15c469a` — M2 M3 M4: freeze one-shot annotation branch and deterministic selection
- `d48d085` — M2 M4 M6: freeze final pilot-selected method before opening held-out regions

## M5 — Unit contract and measured data quality

- `bf9f7e4` — M1 M3 M5 M6: fix terrain minimum and preserve export provenance; prepare blind audit
- `74f59fe` — M1 M5 M6 M7: verify zero metric delta and repaired point provenance

## M6 — Preregistered loss and separate LAS ontology/provenance

- `2b0cfaf` — M4 M6: preregister balanced error and equal abstention cost
- `bf9f7e4` — M1 M3 M5 M6: fix terrain minimum and preserve export provenance; prepare blind audit
- `74f59fe` — M1 M5 M6 M7: verify zero metric delta and repaired point provenance
- `d48d085` — M2 M4 M6: freeze final pilot-selected method before opening held-out regions

## M7 — Defect checks, robustness and sensitivity evidence

- `208ece4` — M4 M7: freeze matched components and bounded sensitivity execution
- `0620ab9` — M1 M4 M7: preregister corrected anchor and all raster sensitivities
- `74f59fe` — M1 M5 M6 M7: verify zero metric delta and repaired point provenance
- `2449ccc` — M3 M4 M7: retain all component outcomes and select only by locked objective

## M8 — Evidence framing, engineering limits and venue recommendation

- No methodological claim was added. The final document-packaging commit supplies the measured runtime limits, unresolved engineering endpoint, replacement text and honest venue recommendation; inspect the branch log for its hash.

