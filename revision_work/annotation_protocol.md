# Independent annotation and pilot-label audit protocol

Protocol drafted September 23, 2026 for reviewer items M2, M3, M5 and M6. This is a protocol, **not completed annotation or an independent accuracy result**. It authorizes producing new prediction-hidden annotation inputs, an ID key, and a sample manifest. It does not authorize the research software or assistant to create, change, adjudicate, or infer reference labels.

## Immutable sources and access order

1. The existing pilot reference file stays byte-identical. Record its existing hash before preparing pilot audit copies; verify it after exporting them. Audit responses are new human-authored records, not edits to that reference file.
2. `holdout.json` and `objective.md` are preregistered and hash-locked. Until Phase 5, do not open, hash, display, summarize, infer, or score any A/B data or cached whole-scene content. Geographic metadata in the preregistered JSON can be read without opening those inputs. Development must stay within the allowed tile-C context.
3. Commit the selected final manifest before consuming `begin_phase5`. Gate creation is exclusive and irreversible, including an annotation-only event or a failed event. A failed or incomplete run must not delete the marker, retry the held-out run, or reuse this protocol for a second scoring pass. The terminal record is appended separately; the original reservation is never overwritten.
4. At Phase 5, if independent human reference annotations or their provenance are absent, use an **annotation-only** workflow. Create prediction-hidden inputs, inventory point counts/identity, and stop. Do not manufacture reference classes, infer accuracy, or select a configuration from the held-out data. Later human annotation and any subsequent scoring require a new explicitly approved protocol.

## Export contract: a new unlabelled input is not a reference

- Export each held-out **scoring** rectangle exactly as preregistered, using half-open XY bounds and all heights. Parent-tile protection applies before Phase 5. Give the new annotation input an explicit name such as `holdout_a_annotation_input.las`; never overwrite an original LAS or reference path.
- In the new annotation copy only, set classification to LAS unclassified (1), clear classification flags, and omit any existing reference or prediction fields, confidence/detection scores, fallback identifiers, error flags, model overlays, class-colored renderings, semantic filenames, or classification look-up VLRs. Audit copies must hide the old reference labels as well as the predictions.
- Use an explicit allowlist of observational fields: original XYZ/scales/offsets, original RGB, intensity and return attributes, plus a new `local_id`. Other attributes require review for semantic leakage before export. Ordinary RGB is observational context, not a prediction. Metadata may preserve verified coordinate information, but unsupported sensor/date/colorization/CRS values are **UNKNOWN - needs provider/PI**.
- Assign unique per-package local IDs 0 through N-1, preserving their integer values exactly. If a GUI path uses a float32 scalar field, every integer must be <=16,777,216; split oversized packages before reaching that limit. Never rely on rounded global IDs exported through a GUI.
- Keep `local_id -> immutable original ID -> source hash` in a separate protected mapping file. This key is for reconciliation and integrity checking, not annotator display. The annotator package must not include an easy join to existing model outputs or original reference labels. Source paths and ID maps remain with the data custodian.
- Save a manifest containing region bounds, coordinate units/known CRS, source hash, export hash, schema, point count, local-ID range and uniqueness, XYZ/RGB correspondence check, extraction code hash, seed where applicable, and preparation date. Hashing a held-out source is allowed only after Phase 5 starts.
- Verify round-trip IDs and coordinates using a synthetic or pilot example before the held-out export. Human annotation should return a CSV keyed by local ID (or a separately returned annotated copy whose IDs are reconciled); never accept row order or nearest-point matching as identity evidence. Reject duplicated, missing, non-integer, out-of-range, or altered IDs. Coordinate and RGB differences require investigation before using a response.

## Independent pilot-label audit sample

The pilot is development data. Sampling is an audit of its current labels, not a way for software to relabel difficult points or improve a measured score.

The eligible universe is the existing reviewed pilot IDs within its fixed 40 x 30 ft bounds. Use review membership only to identify that universe; do not use semantic class, correctness, confidence, model output, or an error map for primary sample selection. Reference class and predictions remain hidden from both annotators until their responses are locked.

Use a deterministic stratified sample with seed **20260923**. The following strata use input geometry/color only and are fixed before human review:

| Input attribute | Strata |
|---|---|
| Height above the independently documented terrain estimate | HAG <2 ft; 2 <= HAG <=6 ft; HAG >6 ft; nonfinite/unknown HAG |
| Original RGB ExG = (2G-R-B)/(R+G+B) | ExG <=0.05; ExG >0.05; invalid/missing color or zero channel sum |
| Geometric/color boundary indicator | Boundary; interior |

Compute the boundary indicator using a fixed 0.5 US survey ft grid over the allowed pilot context, never using semantic labels. A cell is a boundary if one of its eight adjacent cells is empty, the range of their finite maximum elevations exceeds 2 ft, or the range of their finite mean ExG values exceeds 0.05. Any point in such a cell belongs to the boundary stratum. Report the grid origin, whether neighboring context was available, and the count of each triggering condition. These thresholds define audit sampling; they are not claimed to be physical class boundaries.

The cross-product has at most 24 strata. Sample up to **50 unique target points per nonempty stratum** without replacement, for at most 1,200 audit targets. Sort candidate IDs before seeded sampling, so file iteration order cannot change the sample. Census strata containing fewer than 50 points. Record N_h, n_h, and selection probability n_h/N_h for every stratum, including empty strata. Do not replace difficult or ambiguous selected points. Do not add model-error-selected points to the primary sample; any separate targeted diagnostic sample must be explicitly distinguished and excluded from population-rate estimates.

Provide RGB/XYZ context around each target without colorizing the target by its original label or model result. A neutral `audit_target` identifier may highlight the selected point, while neighboring points remain unlabelled context. Context need not be independently labelled. Sampling strata and HAG groups should be withheld from annotators to avoid prompting a semantic interpretation. Closely spaced targets may share a context view; this does not make them statistically independent.

For the eventual audit report, use sampling weights N_h/n_h if estimating disagreement on the reviewed pilot. Report per-stratum sample counts, original-versus-human agreement after response lock, uncertain rate, and disagreements. Do not treat dense points as independent trials or report a naive point-binomial confidence interval. A pilot-only audit cannot estimate between-site generalization. No model score should be recomputed against newly changed references without a separate human-approved decision and a versioned evaluation protocol.

## Human semantic definitions and ambiguity

The preregistered possible semantic taxonomy is grass (3), tree (5), building (6), hard surface (11), and vehicle (64). These are project concepts with a documented LAS export crosswalk, not evidence that LAS vegetation height categories perfectly define plant types. Annotators must use observable object/material context rather than an automatic height threshold.

| Semantic response | Human guidance |
|---|---|
| Grass | Points visibly belonging to grass/herbaceous ground cover. Ground-level position alone is insufficient. Do not force bare soil, leaves/litter, shrubs, or indeterminate low vegetation into this class. |
| Tree | Observed tree leaves, branches or woody structure attributable to a tree, regardless of height. Low branches/trunks are not automatically grass. Ambiguous shrubs or bushes require uncertainty/other rather than an invented rule. |
| Building | Observed building structure, including roof/facade when identifiable. Do not classify an elevated object as building solely from height. |
| Hard surface | Identifiable constructed paved surface such as road, sidewalk or parking pavement. If rock, bare soil, dark vegetation, or debris cannot be distinguished, record uncertainty/other. The current export merges road/sidewalk/parking; annotators may add a separate descriptive note without inventing additional scored classes. |
| Vehicle | An identifiable vehicle point; use object context. Nonvehicle equipment, fences, furniture or an indeterminate small object is not automatically vehicle. |
| Uncertain / unobservable | Evidence is insufficient, color appears misregistered, objects overlap, or the point lies on an unresolved boundary. Record the reason. This is an annotation status, not a model prediction. |
| Other / outside taxonomy | A confidently identified material/object that does not fit the preregistered five classes. Record its description. Do not silently collapse it into a convenient class. |

The fixed output columns are `local_id`, `annotation_status`, `semantic_class`, `reason`, `annotator_id`, and `annotation_time`. Allowed statuses are `confident`, `uncertain`, and `other`. `semantic_class` must be empty for uncertain/other. The exported template starts entirely empty: software does not prepopulate semantic classes, default every point to a class, or assign confidence to the human.

Two human annotators should independently review the same audit targets while blinded to each other's responses, the old labels, and every model prediction. After both responses are locked, a human adjudicator reviews disagreements using raw evidence; preserve both originals, adjudication reasoning, and the final human decision. Do not force consensus where evidence remains ambiguous. If only one reviewer is available, label the audit as single-reviewer and do not claim inter-annotator agreement or independent adjudication.

Independent full-region reference annotation requires the same definitions, prediction blindness, identity preservation and uncertainty logging. Human label provenance must state coverage, annotators, review/adjudication procedure, dates and any grouping tools used. Annotation effort should be measured in human time and reviewed coverage. Training annotators must use examples outside the locked held-out regions before Phase 5.

## Human decisions and unresolved provenance

The PI/provider must supply sensor/platform, acquisition date, colorization workflow and alignment checks, and coordinate/datum provenance if not verifiable from source metadata. Record missing entries verbatim as **UNKNOWN - needs provider/PI**. Do not infer them from filenames or appearance.

An unresolved taxonomy change that alters the research question requires PI approval and a new protocol; retain the original taxonomy and label the affected points uncertain/other in the meantime. Annotation software and the research assistant may prepare packages and reconcile identities but may not supply semantic judgments, edit reference labels, adjudicate disagreements, or silently discard hard cases.

## Deliverables and honest stopping point

The preparation deliverables are prediction-hidden annotation inputs, a blank response template, a separate protected ID mapping, immutable source/export manifests, the deterministic pilot audit sample, and this protocol. They are **not** ground truth. Mark M3 as awaiting independent human responses until those responses and their audit trail exist. If Phase 5 ends at annotation export, record `blocked_no_reference`, leave final held-out accuracy unreported, and state the exact human annotation work needed before a separately authorized future evaluation.
