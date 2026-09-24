"""Generate revision tables and scientific figures directly from saved experiment JSON."""

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "research_paper_documents_and_drafts"
TABLES = DOC / "supporting/revision_experiments_2026-09-23"
FIGS = DOC / "images_and_charts/revision_experiments_2026-09-23"
NAMES = {3: "Grass", 5: "Tree", 6: "Building", 11: "Hard surface", 64: "Vehicle"}


def write_csv(name, rows):
    with (TABLES / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_results():
    from revision.experiments import configurations

    records = {}
    for exp in ["fixed_anchor", "grid_025", "grid_1", "rgb_band_075", "rgb_band_3"]:
        folder = "analysis_full" if exp == "fixed_anchor" else "analysis"
        file = ROOT / f"revision_work/runs/{exp}/{folder}/metrics.json"
        value = json.loads(file.read_text())
        expected = {cfg["id"] for cfg in configurations(exp == "fixed_anchor")}
        if set(value["methods"]) != expected or not value.get("development_only"):
            raise ValueError(f"Incomplete or unexpected experiment methods in {file}")
        for name, m in value["methods"].items():
            matrix = np.asarray(m["confusion_matrix"])
            if matrix.shape != (5, 7) or int(matrix.sum()) != m["scored_points"]:
                raise ValueError(f"Confusion-count inconsistency in {file}: {name}")
            records[exp + "/" + name] = dict(
                m,
                experiment=exp,
                artifact=str(file.relative_to(ROOT)),
                artifact_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            )
    supports = {
        tuple(m["per_class"][str(code)]["support"] for code in [3, 5, 6, 11, 64])
        for m in records.values()
    }
    if len(supports) != 1:
        raise ValueError(
            "Experiment reference supports differ; do not draw a matched comparison"
        )
    return records


def save(fig, name):
    fig.savefig(FIGS / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIGS / (name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def archive_existing(directory, owned_names=None):
    """Preserve prior generated artifacts before a deliberate regeneration."""
    directory.mkdir(parents=True, exist_ok=True)
    old = [path for path in directory.iterdir() if path.is_file() and (owned_names is None or path.name in owned_names)]
    if old:
        archive = (
            directory
            / "archive"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        archive.mkdir(parents=True)
        for path in old:
            shutil.move(str(path), str(archive / path.name))


def percent(value):
    return None if value is None else 100 * value


def confusion_rows(name, metrics, group="overall", subset="all"):
    return [
        {
            "method": name,
            "group": group,
            "subset": subset,
            "reference": code,
            "prediction": pred,
            "count": metrics["confusion_matrix"][i][j],
        }
        for i, code in enumerate(metrics["reference_codes"])
        for j, pred in enumerate(metrics["prediction_outcomes"])
    ]


def class_rows(name, metrics, group="overall", subset="all"):
    return [
        {
            "method": name,
            "group": group,
            "subset": subset,
            "reference": code,
            "class_name": value["name"],
            "present": value["present"],
            "support": value["support"],
            "true_positive": value["true_positive"],
            "false_positive": value["false_positive"],
            "false_negative": value["false_negative"],
            "precision_percent": percent(value["precision"]),
            "recall_percent": percent(value["recall"]),
            "f1_percent": percent(value["f1"]),
            "iou_percent": percent(value["iou"]),
        }
        for code, value in metrics["per_class"].items()
    ]


def method_title(metrics):
    cfg = metrics["configuration"]
    raster = {
        "fixed_anchor": "Fixed raster",
        "grid_025": "0.25 ft grid",
        "grid_1": "1.0 ft grid",
        "rgb_band_075": "0.75 ft RGB slab",
        "rgb_band_3": "3.0 ft RGB slab",
    }[metrics["experiment"]]
    if "pointwise" in cfg:
        system = (
            "Geometry-only vegetation proxy"
            if cfg["pointwise"] == "geometry"
            else "Pointwise geometry + RGB rules"
        )
    else:
        source = "SAM3" if cfg["source"] == "sam3" else "Raster-rule proposals"
        system = f"{source}; constraints {'on' if cfg['constraints'] else 'off'}, smoothing {'on' if cfg['smoothing'] else 'off'}, {cfg['transfer'].replace('_', '-')} transfer"
        extras = [
            f"{label} {cfg[key]}"
            for key, label in [
                ("majority_size", "window"),
                ("bbox", "box fraction"),
                ("band", "transfer band"),
            ]
            if key in cfg
        ]
        if cfg.get("tree_height_check"):
            extras.append("tree-height check")
        if cfg.get("physical_smoothing"):
            extras.append("physical-smoothing check")
        if extras:
            system += "\n" + ", ".join(extras)
    return raster + ": " + system


def main():
    records = load_results()
    archive_existing(TABLES, {"all_methods.csv", "confusion_matrices.csv", "strata_and_routes.csv", "per_class_metrics.csv", "strata_and_routes_confusion_matrices.csv", "strata_and_routes_per_class_metrics.csv", "matched_sam3_effect.csv", "old_vs_new_tables_5_7.csv", "README.md", "results.json", "generated_artifacts.json"})
    archive_existing(FIGS)
    anchor = {
        k.split("/", 1)[1]: v
        for k, v in records.items()
        if v["experiment"] == "fixed_anchor"
    }
    rows = []
    confusions = []
    strata = []
    per_class = []
    subset_confusions = []
    subset_classes = []
    for name, m in records.items():
        rows.append(
            {
                "method": name,
                "points": m["scored_points"],
                "errors": m["error_points"],
                "balanced_error_percent": 100 * m["primary_loss"],
                "accuracy_percent": 100 * m["overall_accuracy"],
                "mean_iou_percent": 100 * m["macro_iou"],
                "unlabelled": m["unlabelled_points"],
                "invalid": m["invalid_points"],
                "coverage_percent": percent(m["label_coverage"]),
                "proposal_score_kind": m.get("proposal_score_kind"),
                "grass_iou_percent": 100 * m["per_class"]["3"]["iou"],
                "tree_iou_percent": 100 * m["per_class"]["5"]["iou"],
                "artifact": m["artifact"],
            }
        )
        confusions.extend(confusion_rows(name, m))
        per_class.extend(class_rows(name, m))
        for group in ["strata", "routes"]:
            for s, a in m[group].items():
                strata.append(
                    {
                        "method": name,
                        "group": group,
                        "subset": s,
                        "points": a["scored_points"],
                        "errors": a["error_points"],
                        "balanced_error_percent": percent(a["primary_loss"]),
                        "accuracy_percent": percent(a["overall_accuracy"]),
                        "mean_iou_percent": percent(a["macro_iou"]),
                        "unlabelled": a["unlabelled_points"],
                        "invalid": a["invalid_points"],
                        "share_percent": percent(a["share_of_reviewed_points"]),
                        "artifact": m["artifact"],
                    }
                )
                subset_confusions.extend(confusion_rows(name, a, group, s))
                subset_classes.extend(class_rows(name, a, group, s))
    write_csv("all_methods.csv", rows)
    write_csv("confusion_matrices.csv", confusions)
    write_csv("strata_and_routes.csv", strata)
    write_csv("per_class_metrics.csv", per_class)
    write_csv("strata_and_routes_confusion_matrices.csv", subset_confusions)
    write_csv("strata_and_routes_per_class_metrics.csv", subset_classes)
    pairs = []
    for c in [0, 1]:
        for s in [0, 1]:
            for transfer in ["naive", "height_aware"]:
                suffix = f"c{c}_s{s}_{transfer}"
                a = anchor["rules_" + suffix]
                b = anchor["sam3_" + suffix]
                pairs.append(
                    {
                        "constraints": c,
                        "smoothing": s,
                        "transfer": transfer,
                        "rule_errors": a["error_points"],
                        "sam3_errors": b["error_points"],
                        "rule_loss_percent": 100 * a["primary_loss"],
                        "sam3_loss_percent": 100 * b["primary_loss"],
                        "sam3_minus_rules_pp": 100
                        * (b["primary_loss"] - a["primary_loss"]),
                        "rule_artifact": a["artifact"],
                        "sam3_artifact": b["artifact"],
                    }
                )
    write_csv("matched_sam3_effect.csv", pairs)
    historical = json.loads(
        (ROOT / "revision_work/historical_comparison.json").read_text()
    )["methods"]
    compare = []
    for old, new in [
        ("baseline", "point_rules"),
        ("pipeline", "sam3_c1_s1_height_aware"),
        ("height_check", "tree_height"),
        ("smoothing_checks", "physical_smoothing"),
        ("both_combined", "both"),
    ]:
        h = historical[old]
        n = anchor[new]
        compare.append(
            {
                "method": old,
                "old_errors": h["error_points"],
                "new_errors": n["error_points"],
                "delta_errors": n["error_points"] - h["error_points"],
                "old_balanced_error_percent": 100
                * h["derived_for_comparison"]["primary_loss"],
                "new_balanced_error_percent": 100 * n["primary_loss"],
                "old_accuracy_percent": 100 * h["overall_accuracy"],
                "new_accuracy_percent": 100 * n["overall_accuracy"],
                "old_mean_iou_percent": 100 * h["macro_iou"],
                "new_mean_iou_percent": 100 * n["macro_iou"],
                "old_unlabelled": h["unlabelled_points"],
                "new_unlabelled": n["unlabelled_points"],
                "historical_source": h["source"]["path"],
                "fresh_source": n["artifact"],
            }
        )
    write_csv("old_vs_new_tables_5_7.csv", compare)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    # All matched pairs, rather than a selected favorable pair.
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    y = np.arange(len(pairs))
    deltas = [a["sam3_minus_rules_pp"] for a in pairs]
    ax.barh(y, deltas, color=["#b55d3a" if d > 0 else "#276c9b" for d in deltas])
    ax.set_yticks(
        y,
        [
            f"C {'on' if a['constraints'] else 'off'} / S {'on' if a['smoothing'] else 'off'} / {a['transfer'].replace('_', ' ')}"
            for a in pairs
        ],
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SAM3 − rules: class-balanced error (percentage points)")
    ax.set_title("Matched mask-source effect on the development pilot")
    ax.invert_yaxis()
    limit = max(max(abs(value) for value in deltas), 0.01)
    ax.set_xlim(min(0, min(deltas)) - 0.15 * limit, max(0, max(deltas)) + 0.22 * limit)
    for i, value in enumerate(deltas):
        ax.text(
            value + (0.025 * limit if value >= 0 else -0.025 * limit),
            i,
            f"{value:+.3f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.012,
        "C: height/color constraints; S: majority smoothing. Negative values favor SAM3.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "01_matched_mask_effect")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    keys = [
        "geometry_only",
        "point_rules",
        "rules_c1_s1_height_aware",
        "sam3_c1_s1_height_aware",
        "tree_height",
        "physical_smoothing",
        "both",
    ]
    titles = [
        "Geometry only (vegetation proxy)",
        "Pointwise geometry + RGB",
        "Raster rules + shared stages",
        "SAM3 + shared stages",
        "SAM3 + tree height check",
        "SAM3 + smoothing checks",
        "SAM3 + both checks",
    ]
    for ax, metric, title in [
        (axs[0], "primary_loss", "Class-balanced error (%) — lower is better"),
        (axs[1], "macro_iou", "Mean tree/grass IoU (%) ↑"),
    ]:
        ax.barh(
            np.arange(len(keys)),
            [anchor[k][metric] * 100 for k in keys],
            color="#3c718f",
        )
        ax.set_xlabel(title)
        ax.set_yticks(np.arange(len(keys)), titles)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        values = [anchor[k][metric] * 100 for k in keys]
        ax.set_xlim(
            0,
            max(values) * 1.2 if metric == "primary_loss" and max(values) > 0 else 110,
        )
        for i, value in enumerate(values):
            ax.text(
                value + ax.get_xlim()[1] * 0.012,
                i,
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )
    axs[0].invert_yaxis()
    fig.suptitle("Development results; the geometry-only proxy has a restricted task")
    fig.tight_layout()
    save(fig, "02_system_comparison")
    # Complete confusion for every experiment, all outcomes, explicitly absent classes.
    with PdfPages(FIGS / "03_all_confusion_matrices.pdf") as pdf:
        for name, m in records.items():
            matrix = np.array(m["confusion_matrix"])
            support = matrix.sum(axis=1, keepdims=True)
            row_percent = np.divide(
                100.0 * matrix,
                support,
                out=np.zeros(matrix.shape, float),
                where=support > 0,
            )
            fig, ax = plt.subplots(figsize=(10, 5.4))
            ax.imshow(row_percent, cmap="Blues", aspect="auto", vmin=0, vmax=100)
            ax.set_xticks(
                range(7),
                [
                    "Grass",
                    "Tree",
                    "Building",
                    "Hard surface",
                    "Vehicle",
                    "Unlabelled",
                    "Invalid",
                ],
            )
            ax.set_yticks(
                range(5),
                [
                    NAMES[k] + (" (absent)" if sum(matrix[i]) == 0 else "")
                    for i, k in enumerate(m["reference_codes"])
                ],
            )
            for i in range(5):
                for j in range(7):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:,}\n({row_percent[i, j]:.2f}%)"
                        if support[i, 0]
                        else "0\nNA",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if row_percent[i, j] > 55 else "black",
                    )
            ax.set(
                xlabel="Prediction",
                ylabel="Reference class (reviewed groups)",
                title=method_title(m)
                + f"\nCounts and row percentages; n={m['scored_points']:,}. Absent classes have no performance estimate.",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            if name == "fixed_anchor/sam3_c1_s1_height_aware":
                fig.savefig(
                    FIGS / "03_anchor_confusion.png", dpi=300, bbox_inches="tight"
                )
                fig.savefig(FIGS / "03_anchor_confusion.pdf", bbox_inches="tight")
            plt.close(fig)
    # Sensitivity curves use fixed anchor and the complete preregistered alternatives.
    groups = [
        (
            "Grid resolution (US survey ft)",
            [0.25, 0.5, 1.0],
            [
                records["grid_025/sam3_c1_s1_height_aware"],
                anchor["sam3_c1_s1_height_aware"],
                records["grid_1/sam3_c1_s1_height_aware"],
            ],
        ),
        (
            "RGB surface band (ft)",
            [0.75, 1.5, 3.0],
            [
                records["rgb_band_075/sam3_c1_s1_height_aware"],
                anchor["sam3_c1_s1_height_aware"],
                records["rgb_band_3/sam3_c1_s1_height_aware"],
            ],
        ),
        (
            "Transfer band (ft)",
            [1.5, 3.0, 6.0],
            [
                anchor["transfer_band_1.5"],
                anchor["sam3_c1_s1_height_aware"],
                anchor["transfer_band_6"],
            ],
        ),
        (
            "Majority window (pixels)",
            [1, 3, 5, 9],
            [
                anchor["majority_1"],
                anchor["majority_3"],
                anchor["sam3_c1_s1_height_aware"],
                anchor["majority_9"],
            ],
        ),
        (
            "Maximum box fraction",
            [0.5, 0.8, 1.0],
            [anchor["bbox_0.5"], anchor["sam3_c1_s1_height_aware"], anchor["bbox_1"]],
        ),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    sensitivity_max = max(
        100 * item["primary_loss"] for _, _, items in groups for item in items
    )
    for ax, (label, x, ms) in zip(axs.flat, groups):
        values = [100 * a["primary_loss"] for a in ms]
        ax.plot(x, values, "o-", color="#3c718f")
        ax.set_xticks(x)
        ax.set_ylim(0, sensitivity_max * 1.2 if sensitivity_max > 0 else 1)
        for xx, value in zip(x, values):
            ax.annotate(
                f"{value:.3f}",
                (xx, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Class-balanced error (%)")
        ax.grid(alpha=0.2)
    axs.flat[-1].axis("off")
    axs.flat[-1].text(
        0.02,
        0.8,
        "One factor at a time.\nOther settings held at the anchor.\nAll scores use the same reviewed IDs.\nDevelopment selection, not validation.",
        va="top",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "04_parameter_sensitivity")
    # Routes and surface/boundary strata, showing sample sizes as well as errors.
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    m = anchor["sam3_c1_s1_height_aware"]
    for ax, kind, names in [
        (axs[0], "routes", ["transfer", "fallback"]),
        (
            axs[1],
            "strata",
            [
                "surface",
                "below_surface",
                "above_surface",
                "missing_surface",
                "boundary",
                "interior",
            ],
        ),
    ]:
        vals = [m[kind][n] for n in names]
        populated = [
            (i, a) for i, a in enumerate(vals) if a["primary_loss"] is not None
        ]
        ax.barh(
            [i for i, _ in populated],
            [100 * a["primary_loss"] for _, a in populated],
            color="#597c63",
        )
        limit = max([100 * a["primary_loss"] for _, a in populated] + [1])
        ax.set_xlim(0, limit * 1.28)
        for i, a in enumerate(vals):
            if a["primary_loss"] is None:
                ax.text(
                    limit * 0.02, i, "NA — no reviewed points", va="center", fontsize=8
                )
            else:
                ax.text(
                    100 * a["primary_loss"] + limit * 0.02,
                    i,
                    f"{100 * a['primary_loss']:.3f}",
                    va="center",
                    fontsize=8,
                )
        ax.set_yticks(
            range(len(names)),
            [
                n.replace("_", " ") + f" (n={a['scored_points']:,})"
                for n, a in zip(names, vals)
            ],
        )
        ax.set_xlabel("Class-balanced error (%)")
        ax.invert_yaxis()
        ax.set_title(kind.capitalize())
    fig.suptitle(
        "SAM3 anchor: empty strata have no accuracy estimate; boundary overlaps surface groups"
    )
    fig.tight_layout()
    save(fig, "05_routes_and_strata")
    # Same deterministic point view across RGB, reference and two prediction outcomes.
    arr = np.load(ROOT / "revision_work/runs/fixed_anchor/analysis/pilot_arrays.npz")
    reviewed = np.isin(arr["reference"], [3, 5])
    reviewed_indices = np.flatnonzero(reviewed)
    sel = np.sort(
        np.random.default_rng(20260923).choice(
            reviewed_indices, size=min(40000, len(reviewed_indices)), replace=False
        )
    )
    xx = arr["x"][sel] - arr["x"].min()
    yy = arr["y"][sel] - arr["y"].min()
    rgb = arr["rgb"][sel].astype(float) / 65535
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    axs[0].scatter(xx, yy, c=rgb, s=0.6, rasterized=True)
    axs[0].set_title("Observed RGB (sampled reviewed points)")
    for ax, key, title in [
        (axs[1], "sam3_c1_s1_height_aware", "SAM3 anchor errors"),
        (axs[2], "geometry_only", "Geometry-only proxy errors"),
    ]:
        prediction = np.load(
            ROOT
            / f"revision_work/runs/fixed_anchor/analysis_full/{key}_predictions.npz"
        )
        if not np.array_equal(prediction["ids"], arr["ids"]):
            raise ValueError("Prediction IDs differ from spatial-view coordinates")
        p = prediction["prediction"]
        wrong = reviewed_indices[
            p[reviewed_indices] != arr["reference"][reviewed_indices]
        ]
        if len(wrong) != anchor[key]["error_points"]:
            raise ValueError("Spatial error counts differ from metrics")
        ax.scatter(xx, yy, c="#c7ccce", s=0.6, rasterized=True)
        ax.scatter(
            arr["x"][wrong] - arr["x"].min(),
            arr["y"][wrong] - arr["y"].min(),
            c="#b53e32",
            s=2,
            rasterized=True,
        )
        ax.set_title(
            title + f"\nAll reviewed-point errors: {anchor[key]['error_points']:,}"
        )
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlabel("Easting from pilot point minimum (ft)")
    axs[0].set_ylabel("Northing from pilot point minimum (ft)")
    fig.tight_layout()
    save(fig, "06_pilot_error_views")
    n_reviewed = anchor["sam3_c1_s1_height_aware"]["scored_points"]
    grass_support = anchor["sam3_c1_s1_height_aware"]["per_class"]["3"]["support"]
    tree_support = anchor["sam3_c1_s1_height_aware"]["per_class"]["5"]["support"]
    captions = f"""# Revision experiment figures and captions

These figures use the same development pilot of {n_reviewed:,} reviewed points: {grass_support:,} grass and {tree_support:,} tree. Reference building, hard-surface and vehicle classes are absent. Scores are conditional on the existing researcher-confirmed groups; independent annotation and held-out accuracy are not established. Nearby points are not independent statistical replicates.

The preregistered primary metric is class-balanced error: compute error fraction within each reference class present, then average those fractions equally. Correct assignments cost 0; wrong, unlabelled and invalid predictions each cost 1. Lower is better. Overall point accuracy and mean per-class IoU are secondary and need not rank methods identically. Within a stratum, only classes present in that stratum enter its macro averages. Empty strata are NA, never zero error.

1. **01_matched_mask_effect.** Eight paired comparisons change proposal source from geometry-plus-RGB raster rules to SAM3 while matching class vocabulary, thresholds, priority, height/color-constraint toggle (C), majority-smoothing toggle (S), and transfer method. Bars show SAM3 minus rule-source class-balanced error, in percentage points; negative favors SAM3 and positive favors rules. All registered pairs are shown. Rule proposals contain binary support; model detection scores are not calibrated probabilities. This controls the downstream stages, not the statistical meaning of these two proposal scales. No uncertainty across independent regions is available.

2. **02_system_comparison.** Class-balanced error and mean grass/tree IoU for selected named systems on the same reviewed identities. The geometry-only control is a restricted vegetation proxy (height above ground >6 ft gives tree, <2 ft gives grass, otherwise abstention), not a classifier for all infrastructure classes. Raster-rule and SAM3 anchors share enabled constraints, smoothing and height-aware transfer. The last three rows reproduce the historical height-check, physical-smoothing and combined definitions on fresh inputs. Direct labels are computed from metric JSON; all other tested configurations remain in the complete tables.

3. **03_all_confusion_matrices / 03_anchor_confusion.** The multipage PDF includes every one of the {len(records)} recorded configurations; the separate image shows the fixed SAM3 anchor. Rows are reference classes; columns are predicted grass, tree, building, hard surface, vehicle, unlabelled and invalid. Each populated row shows counts and within-row percentages, colored on a common 0–100% scale. An absent reference row displays zero counts and NA percentages. Out-of-class predictions and abstentions remain errors. Absence of reference classes does not establish accuracy for those classes.

4. **04_parameter_sensitivity.** One-factor changes in grid resolution, RGB surface slab, point-transfer band, majority window and maximum bounding-box fraction, with all other settings at the registered anchor. Grid and RGB-slab changes use freshly generated rasters and segmentation. Transfer, majority and box-filter changes reuse only the corresponding image's verified raw responses. Lines connect tested settings for readability, not an interpolated performance model. A common zero-based vertical range supports comparison. Grid changes also change the physical footprint of pixel-sized smoothing/inpainting windows; RGB-slab changes also change auxiliary surface ExG/HAG aggregates. These are pipeline-setting sensitivities, not isolated physical effects, independent validation or a general optimum.

5. **05_routes_and_strata.** Fixed SAM3-anchor class-balanced error, with reviewed sample counts, by transfer attempt/fallback and by geometry/reference-defined subsets. Surface means absolute point-to-cell-maximum elevation difference <3 ft; below/above are finite non-surface points and missing surface is reported separately. Boundary points lie within 1 ft Euclidean 3D distance of a reviewed point of another class, or within 1 ft horizontally of the pilot perimeter; unreviewed points do not create class boundaries. Boundary/interior partition the reviewed points and overlap the surface subsets. Transfer attempts include unlabelled surface predictions. Empty subsets explicitly have no accuracy estimate. Pointwise controls have no transfer/fallback stage and are marked unknown/not applicable in the CSVs.

6. **06_pilot_error_views.** XY projections of original 16-bit LAS RGB divided by 65,535 and all error locations for the SAM3 anchor and restricted geometry-only proxy. A fixed seeded sample of {len(sel):,} reviewed points supplies the RGB and gray context in every panel; every reviewed error is plotted in red without subsampling. All heights are projected, so canopy and underlying grass can overlap. Axes are offsets in US survey feet from the pilot point-coordinate minima. Prediction IDs are checked against coordinate IDs and plotted error counts against the complete metric records. This is a spatial display, not an independent registration or label-quality assessment.

Historical-versus-fresh table comparisons retain the September 14 saved-count evidence separately. The fresh tile-C-only preprocessing changes context and grid origin relative to the historical whole-survey route; differences cannot be attributed to SAM3 alone. Historical primary losses are retrospective arithmetic under the new objective, not originally reported primary metrics.
"""
    (FIGS / "captions.md").write_text(captions)
    (TABLES / "README.md").write_text("""# Generated revision experiment tables

Run `.venv/bin/python -m revision.paper_outputs` only after every registered experiment is complete. The generator preserves earlier output in archive/ before regeneration. All numbers come from saved experiment JSON or confusion counts; figures do not supply or alter reference labels.

- all_methods.csv: complete overall results, outcome coverage, score interpretation, and source artifact.
- per_class_metrics.csv: support, TP/FP/FN, precision, recall, F1 and IoU for all configured classes; absent-class scores are blank.
- confusion_matrices.csv: every cell of every complete 5 by 7 matrix, including zeros, unlabelled and invalid outcomes.
- strata_and_routes.csv: every surface/boundary subset and transfer/fallback/unknown route, with denominator, errors, percentages and reviewed-point share.
- strata_and_routes_confusion_matrices.csv and strata_and_routes_per_class_metrics.csv: complete subset evidence, not only aggregate scores.
- matched_sam3_effect.csv: all eight paired mask-source contrasts; a positive SAM3-minus-rules difference means SAM3 is worse under the primary objective.
- old_vs_new_tables_5_7.csv: saved historical evidence versus fresh corrected-anchor methods, with explicitly retrospective historical primary loss and separate artifact links.
- results.json: complete machine-readable source records and comparisons.
- generated_artifacts.json: generator, input and output hashes plus rendering provenance.

Metric columns ending in percent are percentages; differences ending in pp are percentage points. JSON metric records retain fractions in [0,1]. A blank metric denotes undefined/absent data, not zero. Label coverage excludes unlabelled and invalid predictions. All scored points belong to the development pilot; no held-out accuracy is asserted. See the figure captions for the class-balanced objective, conditional reference scope, score calibration boundary, strata overlap and spatial sampling.
""")
    (TABLES / "results.json").write_text(
        json.dumps(
            {"methods": records, "matched_pairs": pairs, "old_vs_new": compare},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    outputs = [*TABLES.glob("*"), *FIGS.glob("*")]
    input_paths = {ROOT / m["artifact"] for m in records.values()}
    input_paths.update(
        [
            ROOT / "revision_work/historical_comparison.json",
            ROOT / "revision_work/runs/fixed_anchor/analysis/pilot_arrays.npz",
            ROOT
            / "revision_work/runs/fixed_anchor/analysis_full/sam3_c1_s1_height_aware_predictions.npz",
            ROOT
            / "revision_work/runs/fixed_anchor/analysis_full/geometry_only_predictions.npz",
        ]
    )
    (TABLES / "generated_artifacts.json").write_text(
        json.dumps(
            {
                "generator_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": {
                    str(path.relative_to(ROOT)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(input_paths)
                },
                "rendering": {
                    "spatial_context_points": len(sel),
                    "spatial_seed": 20260923,
                    "spatial_errors_subsampled": False,
                    "point_ids_checked": True,
                    "confusion_pdf_pages": len(records),
                    "png_dpi": 300,
                    "metrics_from_saved_artifacts_only": True,
                    "method_records": len(records),
                    "matched_pairs": len(pairs),
                },
                "outputs": {
                    str(p.relative_to(DOC)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in outputs
                    if p.is_file() and p.name != "generated_artifacts.json"
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("Generated", len(records), "method records, tables, and figures at", TABLES)


if __name__ == "__main__":
    main()
