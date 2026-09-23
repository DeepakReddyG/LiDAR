# Fixed bounded experiment plan — before fresh pilot inference

All experiments use immutable pilot references and tile-C geometry only. This plan does not authorize accessing A/B; holdout.json and objective.md take precedence.

## Phase 1

Regenerate features, ground, visible-surface image, SAM3 raw detections, fusion and point transfer from tile C. Run the inherited snapshot unchanged, aside from explicit input/output paths and recorded configuration. Compare with the saved July predictions on the same reviewed identities. A bounded crop changes preprocessing context relative to the historical full survey; differences are genuine, not settings to tune away.

## Phase 2 candidates for actual defect checks

Inspect and test before claiming a fix: terrain raster's documented min aggregation uses fmax; stale segmentation caches are keyed only by filename/prompts; point export mixes detection score with an arbitrary fallback value and overwrites source user_data; CRS/units are not machine recorded. Correct code contracts, not target labels. Each demonstrated fix needs synthetic regression coverage and pilot before/after counts, including a documented zero delta where labels are unaffected. Preserve exact starting sources and outputs.

## Phase 3 component matrix

- Pointwise geometry-only vegetation proxy: HAG > 6 ft -> tree, HAG < 2 ft -> grass, otherwise abstain. This is deliberately a restricted tree/grass proxy, not a multiclass asset classifier; it tests the reviewed sample's separability and cannot assess omitted classes.
- Pointwise geometry+RGB: inherited standalone baseline unchanged.
- Matched image/transfer systems: source proposals from either SAM3 or geometry+RGB surface rules. Use the same class vocabulary, priority, constraint toggle, smoothing toggle, point transfer, output records and evaluation. Cross source {rules,SAM3}, constraints {off,on}, smoothing {off,on}, transfer {naive,height-aware}: 16 configurations. Rule proposals use binary support, not calibrated model scores. Report that distinction.
- Separately reproduce the three reported tree-height/physical-smoothing rule experiments on fresh inputs for Table 7 comparison.

## Phase 3 one-factor sensitivities

Anchor: 0.5 ft grid, 1.5 ft RGB band, 3 ft transfer band, majority size 5, maximum bbox fraction 0.8. Fixed alternatives:

- Grid: 0.25, 1.0 ft; recompute all preprocessing and fresh segmentation per altered raster.
- RGB surface band: 0.75, 3.0 ft; recompute raster and fresh segmentation; geometry/terrain unchanged.
- Transfer band: 1.5, 6.0 ft; reuse only the identical image's hashed responses.
- Majority window: 1, 3, 9 pixels; bbox fraction: 0.5, 1.0.

No Cartesian parameter search. No new prompt optimization. Every tested outcome remains in the report. Reuse requires input image, checkpoint, preprocessing and source provenance to match; bbox sweeps reuse raw, unfiltered detections.

## Phase 4 selection

Choose only among the completed registered configurations according to objective.md. If SAM3 or a rule variant is worse, say so. Prefer a simpler candidate on exact numerical ties. Any observed improvement is development-set selection, not independent confirmation. No extra search after Phase 5.

## Phase 5 stopping condition

Freeze a final selected manifest and commit. Consume the one-shot gate. Inspect whether independent human-approved references exist. If absent or provenance unknown, export prediction-hidden annotation inputs and stop final scoring; do not run unlabelled holdouts for suggestive visuals.

## Phase 6

Scripts regenerate all new experiment tables/figures, a results summary with artifact links, and replacement Sections 3–5 in a separate file. Historical paper figures/results remain preserved. Provider metadata and independent annotation remain human responsibilities.
