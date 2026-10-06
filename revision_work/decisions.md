# Revision decisions — 2026-09-23

- Phase 0: branch revision/2026-09-23 starts at exp; pre-existing modifications remain intact. Only explicitly owned files will be staged; no broad git add.
- Holdouts use deterministic central windows of documented A/B extents with the entire parent crops protected. No new content inspection was used for selection. Historical prior image exposure is disclosed.
- The pilot and tile C are the only allowed development geometry. Fresh bounded preprocessing changes context relative to historical whole-survey processing; any difference will be reported, not tuned away.
- Primary metric is class-balanced error with equal wrong/unlabelled cost. It is committed before experiments, separately from the holdout registration.
- Acquisition/platform/colorization/date/CRS facts not established by source metadata remain UNKNOWN - needs provider/PI.
- No reference labels will be created or edited. Annotation packages are tasks for a human annotator, never ground truth manufactured by this run.

- Phase 1 complete: bounded fresh run v3 took 17.01 seconds, scored all 158,618 reviewed IDs with exact XYZ/scales/offsets. New pipeline 5,215 errors; baseline 5,917. Historical scores are retrospective only. Failed integration attempts (checkpoint lexical suffix and scale-aware slicing) are preserved in revision_work/archive; no settings changed to match scores.

- Phase 2: the independent audit found zero changed predictions after terrain/export fixes. All raster, DTM/HAG and raw detections exactly match Phase 1. 155,485 source user_data values are now preserved. These fixes repair contracts, not accuracy. The old filename-only HTTP cache is outside the revision execution path and remains explicitly unsafe; all experiments use fresh hash-bound responses.

- Phase 3 complete: all 32 registered configurations retained (28 anchor comparisons and four fresh-raster sensitivities). SAM3 increases class-balanced error in all eight matched mask-source comparisons. The geometry-only restricted vegetation proxy is the lowest-loss candidate, 1,393 errors and 0.9453296778% balanced error. This is a development-set selection on height-grouped labels, not evidence of multiclass or held-out capability. No parameter search beyond the committed plan was added.
- M3: generated a blind pilot audit with 457 stratified targets and 158,618 context points; no human semantic response exists and no reference labels were created or changed.

- Phase 4: selected fixed_anchor/geometry_only from all 32 completed candidates under the unchanged objective. Frozen final manifest committed as d48d085 before any held-out access. The selected proxy has no positive building/hard-surface/vehicle capability; its selection reflects the limited reviewed classes and is not a new operational multiclass claim.
- Phase 5: one event consumed, status blocked_no_reference. Both locked A/B reference candidates were absent. Prepared blind A (2,215,472 points) and B (85,293 points) inputs; no model inference, reference edits, scores or tuning. The event cannot be reused.
- Phase 6: table/figure regeneration preserves annotation packages and archives only its own generated files; the initial overly broad report archiver moved the pilot package into archive without changing bytes, which was restored to its intended path. A regression test now protects both annotation directories. No annotation was repeated. Final revision suite: 177 passing tests.
