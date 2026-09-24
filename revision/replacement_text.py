"""Generate separate paper Sections 3–5 from completed revision evidence.

Reads registered development JSON records, not point clouds or held-out images.
The original manuscript is never overwritten. Execute only after Phase 5's
terminal annotation/evaluation status exists.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from revision.guard import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "research_paper_documents_and_drafts"
ANCHOR = "sam3_c1_s1_height_aware"
RASTER_RUNS = ["grid_025", "grid_1", "rgb_band_075", "rgb_band_3"]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def link(label, relative):
    # This output is in the canonical folder at the repository root.
    return f"[{label}](../{relative})"


def pc(value, decimals=3):
    return "not estimable" if value is None else f"{100 * value:.{decimals}f}%"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def method_sections(m, metrics, raw):
    p, model, csf = m["parameters"], m["model"], m["csf_parameters"]
    classes = p["CLASSES"]
    prompts = table(["Internal ID", "Class", "Text prompt", "Fusion threshold", "LAS code"], [[key, v["name"], v["prompt"], f"{v['threshold']:.2f}", v["las_code"]] for key, v in classes.items()])
    v = p["VETO"]
    constraints = table(["Class", "Eligibility before smoothing"], [
        ["Tree", f"ExG > {v['tree_exg']:g} OR HAG > {v['tree_hag']:g} ft"],
        ["Grass", f"ExG > {v['grass_exg']:g} AND HAG < {v['grass_hag']:g} ft"],
        ["Building", f"HAG > {v['building_hag']:g} ft"],
        ["Road, sidewalk, parking", f"HAG < {v['pavement_hag']:g} ft"],
        ["Vehicle", f"{v['vehicle_hag'][0]:g} < HAG < {v['vehicle_hag'][1]:g} ft"],
    ])
    support = metrics[ANCHOR]["per_class"]
    text = f"""# Replacement Sections 3–5 — September 23, 2026

Generated from saved manifests and numerical evidence by `revision/replacement_text.py`. This is replacement text for integration after PI review. The September 23 manuscript remains unchanged. Table labels below are local to this replacement and require renumbering during integration.

## 3. Methodology

### 3.1. Reproducible bounded execution and input provenance

The experiments use the already prepared tile-C context, nominally 150 × 150 US survey ft, with scoring restricted to the existing 40 × 30 ft pilot. No full-survey preprocessing arrays, images or segmentation caches are reused. Features, terrain, the image raster, raw segmentation responses, fusion and point transfer are regenerated within the permitted context. Limiting the context changes preprocessing and image content relative to the historical survey-wide run; a fresh result is therefore not expected to reproduce the old scores. The parameter values are fixed before inspecting the new results rather than adjusted to recover historical accuracy.

The {link('fixed-anchor manifest', 'revision_work/manifests/fixed_anchor.json')} records pipeline source commit `{m['source_git_commit']}`, per-file source hashes, input and immutable reference hashes, backend/model identity, package versions, parameters, coordinate-unit assumptions and cache policy. The {link('fresh inherited-method manifest', 'revision_work/manifests/phase1_fresh_v3.json')} preserves the starting implementation separately. The command interface runs each phase from its manifest and validates source, input and protocol identities before processing. Old outputs and failed integration attempts are retained separately; they are not substituted for successful inference.

The tile source is LAS {m['dataset']['header']['las_version']}, point format {m['dataset']['header']['point_format']}. Sensor/platform, acquisition date, provider, colorization procedure and color-to-point alignment quality are **UNKNOWN - needs provider/PI**. The local tile header does not declare a CRS. Distances are explicitly configured in US survey feet (1200/3937 m per foot), including the vertical thresholds; that configuration does not verify the provider's coordinate system or vertical datum. Known conflicting CRS units cause a validation error, and no automatic coordinate conversion is performed. The datum remains **UNKNOWN - needs provider/PI**.

### 3.2. Grid, features and terrain

The anchor raster resolution is {p['GRID_RESOLUTION']:g} US survey ft/pixel. Rows increase southward and columns eastward from the input's northwest header bound. Cell indices are `c=floor((x-x_min)/resolution)` and `r=floor((y_max-y)/resolution)`; the far boundary is clipped to the last cell. This many-to-one mapping locates a point in a pixel but does not identify a vertical layer. Original point identities remain separate from the image grid.

Pointwise Excess Green is `ExG=(2G-R-B)/(R+G+B+10^-6)`. Height above ground is `HAG=z-DTM(r,c)`. Intensity and return attributes are extracted, but do not enter the tested classification rules. All source and reference geometry is preserved. Per-point arrays are index-aligned with the source records and memory mapped where used by the stage implementation.

Terrain preprocessing selects the minimum observed elevation in each {p['DECIMATE_CELL']:g} × {p['DECIMATE_CELL']:g} ft cell. A candidate more than {p['GROUND_OUTLIER_FT']:g} ft below its local 5 × 5-cell median is excluded. The Cloth Simulation Filter uses cloth resolution {csf['cloth_resolution']:g} ft, slope smoothing `{csf['bSloopSmooth']}`, classification threshold {csf['class_threshold']:g}, rigidity {csf['rigidness']}, {csf['interations']} iterations and time step {csf['time_step']:g}. These are the recorded software settings, not independently calibrated terrain parameters. Ground samples are aggregated by minimum elevation into the coarse terrain raster. Missing cells are filled with local means using windows {', '.join(str(x) for x in p['DTM_GAPFILL_WINDOWS'])}; residual gaps receive the minimum available terrain value. Bilinear resizing produces the common-grid terrain surface. Negative HAG is retained rather than silently clipped, and no independent terrain reference is available.

### 3.3. Image formation and pinned SAM3 inference

Each occupied pixel first receives its maximum point elevation `S(r,c)`. RGB and auxiliary grid features average points with `z >= S(r,c)-{p['TOP_SURFACE_FT']:g}` ft. Mean LAS RGB is converted to eight bits by division by {p['RGB_16BIT_TO_8BIT_DIVISOR']:g}. Telea inpainting, radius {p['INPAINT_RADIUS_PX']} pixels, supplies model input at empty pixels; the occupancy mask remains separate because inpainted colors are not surveyed observations. Nominal image tiles are {p['TILE_SIZE']} × {p['TILE_SIZE']} pixels with stride {p['TILE_STRIDE']}; the bounded anchor contains one {raw['width']} × {raw['height']}-pixel image.

SAM3 executes through the vendored local MLX implementation on {raw['run_manifest']['device']['device_name']}. The checkpoint is `{model['repository']}`, snapshot `{model['snapshot']}`, SHA-256 `{model['sha256']}`. The runtime uses Python {m['environment']['python'].split()[0]} and MLX {m['environment']['packages']['mlx']}; the manifest contains the remaining dependency versions and tokenizer/source hashes. The explicit local checkpoint path and offline flags prevent an unrecorded model download. Learned parameter keys and shapes are checked against the checkpoint; deterministic positional/causal-mask buffers are generated by the versioned source. The recorded random seed is {model['seed']}.

The processor converts input to RGB, resizes it to {model['processor_resolution']} × {model['processor_resolution']} using PIL LANCZOS, divides by 255 and applies `(value-0.5)/0.5`. The image encoder runs once per image and is reused across text prompts. A detection score is the sigmoid class logit multiplied by the sigmoid presence logit. Detections with score strictly greater than {model['raw_score_floor']:g} are retained in the raw response. Mask probabilities are resized bilinearly to input dimensions with `align_corners=False` and thresholded at 0.5. Raw masks, boxes and scores are preserved; the downstream maximum box-fraction rule removes boxes occupying more than {p['BBOX_FRAC_MAX']:g} of the input tile.

**Table R1. Prompt vocabulary, thresholds and output mapping.**

{prompts}

For each class, overlapping instance masks are combined by the maximum score at each pixel. Overlapping input tiles use the same maximum operation. These values are detection-derived support scores, not calibrated pointwise confidence. Cached responses are consumed only when image, checkpoint, backend/tokenizer source, preprocessing, prompts and parent-manifest identity agree. The legacy filename-only HTTP cache is excluded from all reported experiments and remains unsuitable for reproducible reuse.

### 3.4. Fusion, surface-aware transfer and exported records

Eligible masks claim initially unlabelled pixels in this fixed priority: {', '.join(p['FUSE_PRIORITY'])}. With constraints enabled, eligibility requires the following height/color rule as well as occupancy. Disabling constraints removes the height/color rule while retaining the same vocabulary, thresholds, priority and initial void exclusion.

**Table R2. Height/color constraints.**

{constraints}

The anchor applies an edge-padded {p['MAJORITY_FILTER_SIZE']} × {p['MAJORITY_FILTER_SIZE']} majority filter. Unlabelled pixels participate in voting; ties select the smaller internal label ID. Default smoothing can propagate a label into a pixel that did not originally pass its rule, so pre-smoothing eligibility is not a guarantee about the final label. The separate tree-height and rule-checked smoothing variants test stricter postprocessing explicitly.

A point receives its pixel label when `abs(z-S(r,c)) < {p['MAP_BACK']['surface_ft']:g}` ft. Other points use the registered fallback: HAG > {p['MAP_BACK']['below_tree_hag']:g} ft gives tree; otherwise ExG > {p['MAP_BACK']['below_grass_exg']:g} gives grass; otherwise hard surface. Missing-surface and above-surface fallbacks are recorded separately from below-surface points. Naive transfer, tested as a control, copies the pixel label at every height without applying these fallback rules.

Export preserves the original `user_data`, coordinates, RGB, identifiers, flags and other source dimensions, scales/offsets and CRS/VLR metadata. Classification is replaced with the declared LAS code. New fields store `internal_class_id` (int16), `prediction_source` (uint8) and `model_score` (float32). The source codes are 0 for an unlabelled surface transfer attempt, 1 for a labelled surface-grid transfer, 2 for a below-surface rule, 3 for a missing-surface rule and 4 for an above-surface rule. Model score is NaN for rule and unlabelled outcomes; zero is retained if smoothing transfers a class with no surviving mask support at that pixel. An output-dimension collision fails rather than overwriting an existing source attribute.

Grass/tree are project semantics exported using LAS codes 3/5; they are not defined merely by those LAS vegetation-height categories. Road, sidewalk and parking retain distinct internal IDs but share LAS code 11. Internal IDs preserve that distinction in the new export. Vehicle uses the project's custom code 64. Uncertain or outside-taxonomy human observations must not be silently converted into one of these classes.

## 4. Reference data and evaluation design

### 4.1. Development references and prospective spatial protection

The immutable pilot references contain {support['3']['support']:,} grass and {support['5']['support']:,} tree points, totaling {metrics[ANCHOR]['scored_points']:,} reviewed points across all heights. They were originally prepared as upper/lower geometric groups and confirmed visually by the researcher. They were not individually labelled or independently adjudicated, and their favorable vertical separation is related to the geometric predictors being tested. No reference labels are created, edited or corrected by this revision.

Before fresh experimentation, the protocol committed deterministic central 40 × 30 ft scoring windows within the previously documented A/B extents and protected each entire parent tile against development access. Their explicit coordinates and immutable hashes are in {link('holdout.json', 'holdout.json')}. Selection used recorded coordinates rather than newly inspected content. A/B had appeared in historical whole-scene figures before this protocol; these are prospective tuning holdouts, not a claim of historically unseen imagery. Tile C and the existing pilot are development data throughout.

The independent label-audit deliverable is a protocol and prediction-hidden sample, not additional reference truth. Sampling uses geometry/color strata, a fixed seed and recorded selection probabilities; original semantic labels and model outputs are hidden from annotators. Independent human responses, ambiguity handling and adjudication remain necessary. Details are in the {link('annotation protocol', 'revision_work/annotation_protocol.md')}.

### 4.2. Preregistered objective and completeness checks

The {link('preregistered objective', 'objective.md')} minimizes class-balanced error loss. Correct predictions cost 0; wrong, invalid and unlabelled outcomes each cost 1. Therefore the wrong-to-unlabelled cost ratio is exactly 1:1. For each represented reference class, point losses are averaged within that class, then class losses are averaged equally. For a future multi-region benchmark, region losses would also be averaged equally. Balanced accuracy is one minus this loss. This choice avoids inventing an asset owner's unknown cost schedule and prevents abstention from masquerading as improved primary performance.

Predictions must cover every expected reviewed identifier exactly once. Missing reviewed records fail completeness validation; they do not vanish through an intersection join. Ordered original IDs, integer XYZ, scales and offsets are checked before scoring. Reference codes 0/1 are ignored independently of predictions. Prediction code 1 remains an unlabelled error, and unsupported prediction codes enter an explicit invalid column.

Secondary results include overall accuracy, per-class precision/recall/F1/IoU, mean IoU, coverage, unlabelled rates and full rectangular confusion counts. Mean IoU averages the reference classes actually present. Building, hard surface and vehicle have no pilot reference support, so their recall and class performance are not estimated; their predicted occurrences remain visible as errors. Even the tree/grass precision and IoU are conditional on this restricted reference population because the pilot cannot reveal false positives on omitted ground-truth classes.

Surface, below-surface, above-surface and missing-surface subsets use the recorded transfer geometry. A separate evaluation-only boundary subset contains reviewed points within 1 ft of the pilot perimeter or within 1 ft in three-dimensional Euclidean distance of a reviewed point of another class. Labels used to define this reporting subset do not enter prediction. Boundary and surface subsets overlap and must not be added together. Transfer/fallback shares count attempted transfers, including unlabelled pixel outcomes; pointwise rules have no such routes.

### 4.3. Matched components, sensitivities and selection

The registered component matrix crosses two proposal sources (SAM3 or geometry+RGB surface rules), constraints on/off, smoothing on/off and naive/height-aware transfer. Each paired source comparison retains the same downstream settings and evaluation IDs. Rule proposals are binary support derived from the standalone surface rules; they do not estimate sidewalk, parking or vehicle separately. Sharing a vocabulary and transfer path does not make the two proposal sources equivalent in representational capacity or calibration.

Two pointwise controls are also reported: the inherited geometry+RGB baseline and a restricted geometry-only tree/grass proxy. The latter assigns tree for HAG > {p['BASELINE']['tree_hag']:g} ft, grass for HAG < {p['BASELINE']['grass_hag']:g} ft, and abstains otherwise. It tests the pilot's separability and is not a full infrastructure classifier. The three earlier tree-height and rule-checked smoothing experiments are repeated on fresh inputs.

One-factor sensitivities vary grid resolution (0.25, 0.5, 1 ft/pixel), RGB surface band (0.75, 1.5, 3 ft), transfer band (1.5, 3, 6 ft), majority window (1, 3, 5, 9 pixels) and maximum box fraction (0.5, 0.8, 1). Changed rasters receive fresh segmentation; downstream-only alternatives reuse only identically hashed raw responses. This is a fixed bounded comparison, not a Cartesian search or prompt optimization. Every outcome, including deterioration, is retained.

Candidate selection follows the primary loss on the same development references. Numerical ties within 10^-12 prefer fewer learned components, then fewer postprocessing stages, then smaller measured resource demand. The selected settings and source identities are frozen before a single protected-data access event. If independently approved references are absent, that event ends with prediction-hidden annotation preparation and no held-out scoring. Dense points are spatially correlated; one development patch cannot supply a between-site variance estimate or justify a point-independent confidence interval.
"""
    return text


def results_section(m, methods, raw, sweeps, final, phase5, final_path, phase5_path):
    history = read("revision_work/historical_comparison.json")["methods"]
    fresh = read("revision_work/runs/phase1_fresh/output/metrics.json")
    audit = read("revision_work/evidence/fix_audit.json")
    anchor, rules, geometry = methods[ANCHOR], methods["rules_c1_s1_height_aware"], methods["geometry_only"]
    main_rows = []
    for label, item in [
        ("Historical saved baseline", history["baseline"]),
        ("Historical saved SAM3 pipeline", history["pipeline"]),
        ("Fresh pointwise geometry+RGB baseline", fresh["baseline"]),
        ("Fresh SAM3 anchor, before fixes", fresh["pipeline"]),
        ("Corrected SAM3 anchor", anchor),
        ("Matched surface-rule anchor", rules),
        ("Restricted geometry-only proxy", geometry),
    ]:
        loss = item.get("primary_loss")
        if loss is None:
            present = [entry["error_loss"] for entry in item["per_class"].values() if entry.get("support", 0) > 0]
            loss = sum(present) / len(present)
        main_rows.append([label, f"{item['error_points']:,}", pc(item['overall_accuracy']), pc(loss), pc(item['macro_iou']), f"{item['unlabelled_points']:,}"])
    main_table = table(["Method / evidence", "Errors", "Accuracy", "Balanced error loss", "Mean IoU", "Unlabelled"], main_rows)
    pairs = []
    for constrained, smooth, transfer in itertools.product([0, 1], [0, 1], ["naive", "height_aware"]):
        suffix = f"c{constrained}_s{smooth}_{transfer}"
        a, b = methods["rules_" + suffix], methods["sam3_" + suffix]
        pairs.append({"constraints": constrained, "smoothing": smooth, "transfer": transfer, "rules": a, "sam3": b, "delta": b['primary_loss'] - a['primary_loss']})
    worse = sum(pair["delta"] > 1e-12 for pair in pairs)
    pair_table = table(["Constraints", "Smoothing", "Transfer", "Rule loss", "SAM3 loss", "SAM3 − rules (pp)"], [["on" if pair['constraints'] else "off", "on" if pair['smoothing'] else "off", pair['transfer'].replace('_', '-'), pc(pair['rules']['primary_loss']), pc(pair['sam3']['primary_loss']), f"{100 * pair['delta']:+.3f}"] for pair in pairs])
    sensitivity_rows = [["Anchor", f"{anchor['error_points']:,}", pc(anchor['primary_loss']), pc(anchor['macro_iou']), f"{anchor['unlabelled_points']:,}"]]
    sensitivity_labels = {"grid_025": "Grid 0.25 ft/pixel", "grid_1": "Grid 1 ft/pixel", "rgb_band_075": "RGB band 0.75 ft", "rgb_band_3": "RGB band 3 ft"}
    for name, item in sweeps.items():
        sensitivity_rows.append([sensitivity_labels[name], f"{item['error_points']:,}", pc(item['primary_loss']), pc(item['macro_iou']), f"{item['unlabelled_points']:,}"])
    for name in ["transfer_band_1.5", "transfer_band_6", "majority_1", "majority_3", "majority_9", "bbox_0.5", "bbox_1"]:
        item = methods[name]
        sensitivity_rows.append([name.replace('_', ' '), f"{item['error_points']:,}", pc(item['primary_loss']), pc(item['macro_iou']), f"{item['unlabelled_points']:,}"])
    sensitivity_table = table(["One-factor setting", "Errors", "Balanced error loss", "Mean IoU", "Unlabelled"], sensitivity_rows)
    class_rows = []
    for label, item in [("SAM3 anchor", anchor), ("Surface-rule anchor", rules), ("Geometry-only proxy", geometry)]:
        for code in ["3", "5"]:
            cls = item["per_class"][code]
            class_rows.append([label, cls['name'], f"{cls['support']:,}", pc(cls['precision']), pc(cls['recall']), pc(cls['f1']), pc(cls['iou'])])
    class_table = table(["Method", "Reference class", "Support", "Precision", "Recall", "F1", "IoU"], class_rows)
    strata_table = table(["SAM3 anchor subset", "Reviewed points", "Share", "Errors", "Balanced error loss", "Mean IoU"], [[name.replace('_', ' '), f"{item['scored_points']:,}", pc(item['share_of_reviewed_points']), f"{item['error_points']:,}", pc(item['primary_loss']), pc(item['macro_iou'])] for name, item in anchor['strata'].items()])
    routes = anchor['routes']
    repeat_rows = []
    historical_variant_names = {"tree_height": "height_check", "physical_smoothing": "smoothing_checks", "both": "both_combined"}
    for new_name, old_name in historical_variant_names.items():
        old, new = history[old_name], methods[new_name]
        repeat_rows.append([new_name.replace('_', ' '), f"{old['error_points']:,}", f"{new['error_points']:,}", f"{new['error_points']-anchor['error_points']:+,}", pc(new['primary_loss']), pc(new['macro_iou'])])
    repeated_table = table(["Variant", "Historical errors", "Fresh errors", "Fresh − anchor errors", "Fresh balanced loss", "Fresh mean IoU"], repeat_rows)
    cm_rows = []
    for code in [3, 5]:
        row = anchor['reference_codes'].index(code)
        cm_rows.append(["Grass" if code == 3 else "Tree", *[f"{value:,}" for value in anchor['confusion_matrix'][row]]])
    confusion = table(["Reference", "Grass", "Tree", "Building", "Hard surface", "Vehicle", "Unlabelled", "Invalid"], cm_rows)
    failed_scores = [pair['delta'] for pair in pairs]
    phase5_status = phase5.get("status")
    if phase5_status != "blocked_no_reference":
        raise ValueError("This replacement generator requires the verified annotation-only terminal status; a scored benchmark needs explicit results integration")
    selected = final['selected_candidate']
    selected_loss = final['selected_primary_loss']
    text = f"""

## 5. Development results and limits of confirmation

### 5.1. Fresh evidence supersedes historical performance attribution

**SAM3 does not improve the tested matched pipeline on this pilot:** its primary loss is worse than the corresponding surface-rule proposal system in {worse} of {len(pairs)} paired comparisons. This is evidence about the registered systems on one development patch, not a conclusion that SAM3 fails on infrastructure scenes generally.

The historical saved pipeline had {history['pipeline']['error_points']:,} errors; the bounded fresh inherited pipeline has {fresh['pipeline']['error_points']:,}, a change of {fresh['pipeline']['error_points']-history['pipeline']['error_points']:+,}. The historical baseline had {history['baseline']['error_points']:,} errors, compared with {fresh['baseline']['error_points']:,} for the fresh baseline. New results use freshly computed tile-C terrain/image context and an identified checkpoint and source, whereas complete historical inference provenance is unavailable. The difference cannot be assigned uniquely to a specific historical code or model change. Historical scores remain retrospective evidence and do not replace the fresh results.

**Table R3. Historical and fresh scores on the same reviewed development identities.** Historical loss is reconstructed from saved class-level errors using the newly preregistered objective; it was not the original selection criterion.

{main_table}

The fresh SAM3 pipeline improves on the pointwise geometry+RGB baseline by {fresh['baseline']['error_points']-anchor['error_points']:,} errors, but that comparison includes differences in rasterization, rules, smoothing and transfer. The matched comparison reverses this interpretation: the SAM3 anchor makes {anchor['error_points']-rules['error_points']:,} more errors than the surface-rule anchor, with a {100*(anchor['primary_loss']-rules['primary_loss']):.3f}-percentage-point higher primary loss. Thus the complete-system comparison alone would overstate the image model's contribution. Sources: {link('fresh inherited results', 'revision_work/runs/phase1_fresh/output/metrics.json')}, {link('historical comparison', 'revision_work/historical_comparison.json')}, and {link('complete component results', 'revision_work/runs/fixed_anchor/analysis_full/metrics.json')}.

### 5.2. Controlled attribution and restricted geometry control

**Table R4. Paired proposal-source comparisons.** Positive differences favor surface-rule proposals. The same downstream choices and reference IDs are retained within each row.

{pair_table}

Across these paired settings, the SAM3 loss difference ranges from {100*min(failed_scores):+.3f} to {100*max(failed_scores):+.3f} percentage points. The comparison tests these sources within this workflow; it does not equate binary rule-support values with neural-score calibration or establish performance on classes absent from the pilot.

The restricted height-only proxy has {geometry['error_points']:,} errors and primary loss {pc(geometry['primary_loss'])}. Its favorable score is consistent with the pilot's deliberately clear vertical separation and geometrically grouped reference construction. It does not demonstrate that geometry alone distinguishes grass from bare ground, shrubs, roofs, vehicles or other omitted categories. Adding RGB is not supported as an improvement on this particular reference sample; independent mixed-class regions are required before generalizing that observation.

**Table R5. Per-class results within the reviewed reference population.**

{class_table}

### 5.3. Error destinations, transfer paths and spatial subsets

**Table R6. Fresh SAM3-anchor confusion counts.** Absent reference rows for building, hard surface and vehicle are omitted here but retained as zero-support rows in the machine-readable rectangular matrices. No performance is imputed to those reference classes.

{confusion}

For the SAM3 anchor, {routes['transfer']['scored_points']:,} reviewed points ({pc(routes['transfer']['share_of_reviewed_points'])}) enter the transfer path, which contributes {routes['transfer']['error_points']:,} errors, including {routes['transfer']['unlabelled_points']:,} unlabelled outcomes. The fallback path contains {routes['fallback']['scored_points']:,} points ({pc(routes['fallback']['share_of_reviewed_points'])}) and contributes {routes['fallback']['error_points']:,} errors. The fallback route is identical between matched proposal sources when their grid/surface settings agree, so those errors cannot be credited to or blamed on the selected image proposal source.

**Table R7. SAM3-anchor subset results.** Boundary/interior and surface subsets are different, overlapping partitions. An empty subset has no estimated metric.

{strata_table}

The audit finds {audit['export']['zero_score_surface_transfers']:,} labelled surface transfers with zero surviving-mask score after smoothing. Those records demonstrate why a smoothed semantic label and a detection score must be stored separately. The score should not be interpreted as the probability that the final point label is correct.

### 5.4. Sensitivity and the repeated historical rule variants

Grid changes also alter the physical footprint of pixel-sized smoothing and inpainting windows. RGB-slab changes affect auxiliary surface ExG/HAG aggregates as well as image appearance. These are sensitivities of pipeline settings, not perfectly isolated physical effects.

**Table R8. All registered one-factor sensitivity outcomes.** Parameter alternatives that worsen the primary loss are retained alongside improvements. The anchor is repeated for context, and these are development comparisons rather than independent tests.

{sensitivity_table}

Each altered grid or RGB-band setting has a separate manifest and fresh raw model response: {', '.join(link(name, f'revision_work/manifests/{name}.json') for name in RASTER_RUNS)}. Downstream-only alternatives use the verified anchor image/response. These results cover the specified one-factor settings; they do not establish an optimum across interacting parameter combinations or across sites.

**Table R9. Historical rule variants repeated with fresh inputs.** The relevant comparison for each fresh variant is the fresh anchor, not its old survey-context score.

{repeated_table}

### 5.5. Reproducibility repairs, resource measurement and final stopping point

The terrain aggregation correction and export-field repair leave {audit['pilot_arrays']['prediction']['changed_values']:,} point predictions changed between the fresh inherited and corrected anchors; both have {audit['before_after']['after']['error_points']:,} errors. The derived terrain, HAG, image raster and raw detections match exactly in this before/after audit. The minimum-aggregation bug is exposed by a synthetic duplicate-cell test; one decimated candidate per cell makes it inactive in this pilot. This is a correctness repair with no demonstrated accuracy gain.

The old export overwrote {audit['export']['source_user_data_overwritten_before']:,} source `user_data` values. The corrected export overwrites {audit['export']['source_user_data_overwritten_after']:,}, preserves all original fields except the intended classification change, and adds {audit['export']['added_bytes_per_point']} bytes per point for explicit output provenance. All recorded route/score/class consistency checks pass. Cache-identity and unit checks validate the execution contract; they are not standalone classifier improvements. The full record is the {link('independent fix audit', 'revision_work/evidence/fix_audit.json')}.

The measured anchor worker processes its image and prompts in {raw['processing_time_ms']/1000:.3f} s, with model loading recorded separately as {raw['run_manifest']['model_load_seconds']:.3f} s and peak MLX allocation {raw['peak_memory_bytes']/2**30:.3f} GiB. This single bounded measurement is neither a runtime distribution nor a whole-survey cost estimate, and excludes other pipeline stages. It does not establish processing throughput for a DOT workload or human correction cost.

The frozen development selection is `{selected}`, with primary loss {pc(selected_loss)}. {final.get('scope_warning', '')} Selection uses the pilot and must therefore be described as development selection rather than independent confirmation. The {link('final manifest', final_path)} was frozen before the single Phase 5 event. That event ended with status `{phase5_status}`: independently approved held-out reference labels were unavailable, so preparation stopped at prediction-hidden annotation packages. No held-out accuracy, inference-based qualitative success or final generalization claim is reported. The terminal evidence is the {link('Phase 5 completion record', phase5_path)}.

The supported contribution is now a reproducible bounded comparison with controlled negative evidence about SAM3's incremental benefit on this selected tree/grass sample, together with explicit transfer/export diagnostics. Independent reference annotation, acquisition/colorization/CRS provenance, representative multiclass evaluation and an asset-owner task remain required before claiming operational semantic asset-management performance.
"""
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-record", required=True, help="Completed terminal metadata JSON, not raw protected data")
    parser.add_argument("--final-manifest", required=True)
    parser.add_argument("--output", default="replacement_sections_3_5_2026-09-23.md")
    args = parser.parse_args(argv)
    output = DOC / args.output
    if output.exists():
        raise FileExistsError("Preserve the existing replacement text; choose a new output")
    m = read("revision_work/manifests/fixed_anchor.json")
    methods = read("revision_work/runs/fixed_anchor/analysis_full/metrics.json")["methods"]
    raw = read("revision_work/runs/fixed_anchor/masks/raw/tile_r0_c0.json")
    # All registered raster results and the final Phase 5 outcome must exist;
    # partial runs are not converted into apparent complete paper results.
    sweeps = {name: read(f"revision_work/runs/{name}/analysis/metrics.json")["methods"][ANCHOR] for name in RASTER_RUNS}
    phase5 = read(args.phase5_record)
    final = read(args.final_manifest)
    text = method_sections(m, methods, raw) + results_section(m, methods, raw, sweeps, final, phase5, args.final_manifest, args.phase5_record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(text)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), "words": len(text.split())}))


if __name__ == "__main__":
    main()
