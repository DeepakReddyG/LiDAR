# Revision decisions — 2026-09-23

- Phase 0: branch revision/2026-09-23 starts at exp; pre-existing modifications remain intact. Only explicitly owned files will be staged; no broad git add.
- Holdouts use deterministic central windows of documented A/B extents with the entire parent crops protected. No new content inspection was used for selection. Historical prior image exposure is disclosed.
- The pilot and tile C are the only allowed development geometry. Fresh bounded preprocessing changes context relative to historical whole-survey processing; any difference will be reported, not tuned away.
- Primary metric is class-balanced error with equal wrong/unlabelled cost. It is committed before experiments, separately from the holdout registration.
- Acquisition/platform/colorization/date/CRS facts not established by source metadata remain UNKNOWN - needs provider/PI.
- No reference labels will be created or edited. Annotation packages are tasks for a human annotator, never ground truth manufactured by this run.

- Phase 1 complete: bounded fresh run v3 took 17.01 seconds, scored all 158,618 reviewed IDs with exact XYZ/scales/offsets. New pipeline 5,215 errors; baseline 5,917. Historical scores are retrospective only. Failed integration attempts (checkpoint lexical suffix and scale-aware slicing) are preserved in revision_work/archive; no settings changed to match scores.

- Phase 2: the independent audit found zero changed predictions after terrain/export fixes. All raster, DTM/HAG and raw detections exactly match Phase 1. 155,485 source user_data values are now preserved. These fixes repair contracts, not accuracy. The old filename-only HTTP cache is outside the revision execution path and remains explicitly unsafe; all experiments use fresh hash-bound responses.
