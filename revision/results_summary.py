"""Write the revision evidence summary from locked manifests and measured artifacts."""
import csv
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'research_paper_documents_and_drafts'
EVIDENCE=ROOT/'revision_work/evidence'


def main():
    data=json.loads((DOC/'supporting/revision_experiments_2026-09-23/results.json').read_text())
    methods=data['methods'];pairs=data['matched_pairs'];oldnew=data['old_vs_new']
    final=json.loads((ROOT/'revision_work/manifests/final_manifest.json').read_text())
    completion=json.loads((ROOT/'revision_work/phase5_gate.json.completion.json').read_text())
    gate=json.loads((ROOT/'revision_work/phase5_gate.json').read_text())
    audit=json.loads((EVIDENCE/'fix_audit.json').read_text())
    negative=all(a['sam3_minus_rules_pp']>0 for a in pairs)
    opening=('SAM3 added no measured benefit on this development pilot: it increased the preregistered class-balanced error in all eight matched comparisons.' if negative else 'SAM3 produced mixed matched effects on this development pilot; all paired outcomes are reported below.')
    anchor=methods['fixed_anchor/sam3_c1_s1_height_aware'];rules=methods['fixed_anchor/rules_c1_s1_height_aware'];geo=methods['fixed_anchor/geometry_only']
    selected=methods[final['selected_candidate']]
    lines=[opening,'', '# Evidence produced — September 23, 2026','',
      'This is the result of new local experiments on branch `revision/2026-09-23`, following the simulated reviewer report. The original manuscript, historical outputs and reference labels are preserved. This document is not professor feedback or an editorial decision.','',
      '## Main findings','',
      f"- The matched anchor uses identical constraints, smoothing and height-aware transfer. SAM3 has **{anchor['error_points']:,} errors**, **{100*anchor['primary_loss']:.6f}% balanced error**, **{100*anchor['overall_accuracy']:.4f}% accuracy** and **{100*anchor['macro_iou']:.4f}% mean tree/grass IoU**. Rule proposals have **{rules['error_points']:,} errors**, **{100*rules['primary_loss']:.6f}% balanced error**, and **{100*rules['overall_accuracy']:.4f}% accuracy**. [All paired effects](supporting/revision_experiments_2026-09-23/matched_sam3_effect.csv).",
      f"- The geometry-only vegetation proxy has **{geo['error_points']:,} errors**, **{100*geo['primary_loss']:.6f}% balanced error**, **{100*geo['overall_accuracy']:.4f}% accuracy** and **{100*geo['macro_iou']:.4f}% mean IoU**. It wins the committed development objective. It predicts grass/tree/abstention only; this deliberately height-separated, geometrically grouped reference favors such a proxy. These numbers do not validate it as a multiclass system. [All methods](supporting/revision_experiments_2026-09-23/all_methods.csv).",
      '- The original two-system comparison does not isolate SAM3. Once the mask source is matched, every measured SAM3 effect is adverse. This does not establish that SAM3 is useless on other scenes, prompts, viewpoints or independently annotated data.',
      f"- The selected configuration is `{final['selected_candidate']}` under the preregistered equal-cost loss. Selection is explicitly development work. The frozen manifest predates held-out access: commit `{gate['final_commit']}`. [Frozen final manifest](../revision_work/manifests/final_manifest.json).",
      f"- Phase 5 status: **{completion['status']}**. No held-out predictions, accuracy, confidence intervals or adaptation were produced. The one-shot access event is consumed. A/B annotation packages are preparation tasks, not reference labels. [Gate](../revision_work/phase5_gate.json) and [completion](../revision_work/phase5_gate.json.completion.json).",'',
      '## Old versus new: Tables 5–7','',
      'Historical scores used saved July predictions. Fresh scores use the same 158,618 reviewed IDs but bounded tile-C preprocessing. Context, raster origin and terrain differ from the historical survey run; score changes cannot be attributed solely to a bug fix or the model. No settings were adjusted to recover old numbers.','',
      '| Method | Old errors | Fresh errors | Old accuracy | Fresh accuracy | Old mIoU | Fresh mIoU |','|---|---:|---:|---:|---:|---:|---:|']
    for a in oldnew:lines.append(f"| {a['method']} | {a['old_errors']:,} | {a['new_errors']:,} | {a['old_accuracy_percent']:.2f}% | {a['new_accuracy_percent']:.2f}% | {a['old_mean_iou_percent']:.2f}% | {a['new_mean_iou_percent']:.2f}% |")
    lines += ['', '[Machine-generated table](supporting/revision_experiments_2026-09-23/old_vs_new_tables_5_7.csv). Full confusion matrices and per-class precision/recall/F1/IoU are in the [generated table index](supporting/revision_experiments_2026-09-23/README.md). An absent reference class is not assigned an artificial perfect or zero score.','',
      '## What failed and what changed','',
      f"- The fresh tree-height check produces {methods['fixed_anchor/tree_height']['error_points']:,} errors; physical smoothing and combined checks each produce {methods['fixed_anchor/physical_smoothing']['error_points']:,}. Neither improves the anchor under the locked objective.",
      f"- The 0.5 box-fraction rejection threshold produces {methods['fixed_anchor/bbox_0.5']['error_points']:,} errors and {100*methods['fixed_anchor/bbox_0.5']['primary_loss']:.6f}% balanced loss. This is a major robustness failure on the development patch, not a number to omit. [Sensitivity plots](images_and_charts/revision_experiments_2026-09-23/04_parameter_sensitivity.pdf).",
      '- Grid changes also alter the physical footprint of pixel-sized smoothing/inpainting windows; RGB-slab changes affect the surface ExG/HAG aggregates as well as image appearance. These test pipeline settings, not perfectly isolated physical mechanisms.',
      '- Naive column transfer is substantially worse than height-aware transfer for both proposal sources. The comparison measures the whole transfer-plus-fallback design; it is not an isolated causal estimate of geometric occlusion reasoning.',
      '- The DTM minimum-aggregation fix changes zero pilot labels because the old and corrected pilot DTM/HAG arrays are identical. Export now preserves all source fields except intended classification, adds separate score/source/internal-class fields, and changes zero predictions. It prevents 155,485 source user_data values being overwritten. [Independent fix audit](../revision_work/evidence/fix_audit.json); [changelog](../CHANGELOG.md).',
      '- The revision runner rejects missing IDs, changed geometry/scales, altered manifests and incomplete/stale raw responses. The old HTTP filename cache is outside this verified route and remains unsafe; do not treat its historical settings as recovered.',
      '- Model inference was measured on the local MLX/Metal backend, not PyTorch MPS. Corrected bounded runs took about 16–17 seconds including preprocessing, inference, scoring and artifact writing. Reported peak MLX allocation was about 8.5 GB; this is not whole-process peak RSS, cold-start cost, full-survey scalability or annotation labor. [Run timings](../revision_work/evidence/run_timings.json).','',
      '## Claims now supported, and limits','',
      '| Claim | Evidence | Boundary |','|---|---|---|',
      '| A frozen current implementation produces the reported fresh pilot results. | [Run manifests](../revision_work/manifests/fixed_anchor.json), hashes, fresh outputs and independent arithmetic audit. | Local reproducibility; unknown July provenance is not recovered; raw-data redistribution/access is unresolved. |',
      '| SAM3 is worse than rule proposals in the eight registered matched pilot comparisons. | [Paired effects](supporting/revision_experiments_2026-09-23/matched_sam3_effect.csv). | One development patch and fixed prompts/backend; not a general model ranking. |',
      '| Transfer and heuristic settings materially affect this pilot. | [All configurations](supporting/revision_experiments_2026-09-23/all_methods.csv), full strata/routes and sensitivities. | No terrain ground-control validation or independent-region uncertainty. |',
      '| Export preserves identity, geometry, metadata and separate score provenance. | [Fix audit](../revision_work/evidence/fix_audit.json), synthetic LAS roundtrip tests. | A score remains uncalibrated and may be zero after smoothing. |',
      '| Density and RGB storage characteristics have been measured in C. | [Data-quality report](../revision_work/evidence/data_quality.md). | Sensor/platform/date/colorization/registration/CRS/vertical datum remain UNKNOWN - needs provider/PI. |',
      '| Blinded independent annotation can begin. | [Pilot annotator package](supporting/revision_experiments_2026-09-23/pilot_label_audit/annotator), [held-out annotator packages](supporting/revision_experiments_2026-09-23/holdout_annotation/annotator), [protocol](../revision_work/annotation_protocol.md). | No human labels or audit have been completed. Keep private mappings and predictions from annotators. |','',
      '## Reviewer items and exact remaining human work','',
      '| Item | Status | What remains |','|---|---|---|',
      '| M1 reproducibility | Resolved for the new local pilot execution; historical/data-sharing portions remain open. | Provider/PI must settle raw-data access or redistribution and checkpoint/data availability to another lab. Do not present the July run as reconstructed. |',
      '| M2 generalization | Blocked on independent references. Prospective holdout boundaries and one-shot gate are implemented. | Independently annotate both fixed A/B rectangles; supply annotators, dates, coverage, uncertainty and adjudication provenance. Future scoring needs a new explicitly approved protocol; never reuse the consumed gate. |',
      '| M3 label validity | Package/protocol complete; semantic audit blocked on humans. | Two blinded humans review the 457 pilot targets, retain independent responses, then a human adjudicator records disagreements and uncertainty. Do not overwrite the original pilot reference. |',
      '| M4 SAM3 attribution | Resolved within the registered pilot matrix; broader comparison remains partial. | Independent multi-region evaluation and a credible conventional learned point-feature comparator would be needed for a broader performance claim. |',
      '| M5 data/RGB provenance | Descriptive measurements and unit contract complete; acquisition/registration claims blocked. | Obtain sensor/platform, date, coordinate and vertical datums, acquisition/overlap information, colorization workflow and alignment checks from the provider/PI. |',
      '| M6 objective/export ontology | Resolved for this experiment; operational utility remains open. | Confirm asset-owner error/abstention costs before any deployment claim; use the blind protocol to record uncertain/out-of-taxonomy objects rather than relabel them automatically. |',
      '| M7 robustness | Partially resolved. All requested bounded component and parameter sweeps are complete. | Independent terrain/profile checks, outlier tests and broader scenes are needed before claiming robustness. |',
      '| M8 novelty/engineering value | Unresolved as a journal contribution. Local runtime is measured. | Choose an asset-owner decision and measure relevant omission/commission, correction time and cost with independent evidence. Changing that research question requires PI/user direction. |','',
      '## Honest venue recommendation','',
      'The present evidence supports an internal technical report/preprint and a narrowly framed reproducibility/negative-results pilot submission to an appropriate remote-sensing workshop. An ISPRS event publishing technical contributions in the Archives is a plausible format, subject to that event’s scope and review; this is not an acceptance prediction. ISPRS distinguishes [Archives technical proceedings](https://isprs-archives.copernicus.org/) from [Annals full-paper peer review](https://isprs-annals.copernicus.org/).',
      'I would not submit this as a full research article to Automation in Construction, ASCE Journal of Computing in Civil Engineering, or ISPRS Journal of Photogrammetry and Remote Sensing yet. The obstacle is independent semantic validation, generalization and an engineering/research contribution, not polishing the figures. The [ASCE journal scope](https://ascelibrary.org/page/jccee5/editorialboard) includes sensing and infrastructure computing; topic fit alone does not resolve those evidentiary gaps.','',
      '## Deliverables and commits','',
      '- [Separate replacement Sections 3–5](replacement_sections_3_5_2026-09-23.md). The original manuscript is unchanged.',
      '- [Scientific figures and captions](images_and_charts/revision_experiments_2026-09-23/captions.md).',
      '- [Run/reproduction instructions](../revision_work/README.md).',
      '- [Commit history grouped by review item](../revision_work/commit_report.md).',
      '- No messages or materials were sent to the professor, and no model was run on the full 411M-point survey.']
    target=DOC/'results_summary.md'
    if target.exists():
        from datetime import datetime, timezone
        archive=DOC/'supporting/revision_experiments_2026-09-23/archive'
        archive.mkdir(parents=True,exist_ok=True)
        target.rename(archive/('results_summary_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.md'))
    target.write_text('\n'.join(lines)+'\n')
    print(DOC/'results_summary.md')

if __name__=='__main__':main()
