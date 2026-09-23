# Preregistered objective — 2026-09-23

## Primary metric

Minimize **class-balanced error loss** on reviewed reference points: for each reference class present in a region, average point loss; then average those class losses equally. For multiple final regions, average region losses equally. Report balanced accuracy as 1 minus this loss. On the development pilot the reference classes are grass (3) and tree (5). The fixed possible reference taxonomy for later independent annotation is grass (3), tree (5), building (6), hard surface (11), and vehicle (64); an absent reference class is reported absent, not assigned an artificial zero score. Unknown/ambiguous reference points are excluded only by an independently supplied protocol, never by predictions.

Point cost: correct label = 0; wrong label = 1; unlabelled/abstained prediction = 1. Thus the relative cost of unlabelled versus wrong is exactly 1:1. A missing prediction record on any expected reviewed point must fail completeness validation, not disappear from scoring. An invalid prediction code maps to a separate invalid outcome with loss 1.

## Reasoning and secondary metrics

This descriptive semantic-labeling question has no established asset-owner error-cost schedule. Equal wrong/abstention cost avoids inventing one and prevents increased abstention from appearing to improve the primary outcome. Equal class weighting limits domination by sampling density/class abundance. This is a research comparison, not a claimed deployment utility function. Per-class precision, recall, F1, IoU, macro IoU, overall accuracy, full rectangular confusion counts, abstention/invalid rates, and sample counts are mandatory secondary outcomes. They are reported even when their rankings disagree with the primary loss. No metric will be selected after results are seen.

## Selection and statistical scope

The existing pilot is development data. Rank candidates only by the primary loss computed on exactly the same immutable reference IDs. Numerical ties within 1e-12 prefer fewer learned components, then fewer rule/postprocessing stages, then the smaller measured resource demand. Do not select a threshold on held-out data. A candidate must pass record/geometry integrity and holdout guards. This protocol does not justify a point-independent significance test; one spatial region cannot estimate between-site uncertainty. Final evaluation remains descriptive until enough independently annotated regions exist.

## Phase 5

Freeze selected configuration and source hashes in a committed final manifest before opening the held-out regions. Consume the one-shot gate once. If human-approved reference annotations are absent or their provenance is unknown, export annotation packages with predictions hidden and stop; no held-out score or adaptation. Later annotation and any future evaluation require a new explicit protocol, not silently rerunning this gate.
