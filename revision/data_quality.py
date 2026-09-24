"""Describe bounded development input quality without predictions or labels.

Only a completed tile-C run's input.las, HAG, DTM and grid metadata are opened.
The report measures distributions and missing support; it does not validate
terrain accuracy, color registration, sensor provenance or independent samples.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import laspy
import numpy as np

from revision.guard import HoldoutGuard, ProtocolViolation, sha256_file
from revision.units import UNKNOWN, validate_coordinate_units


def _quantiles(values):
    values = np.asarray(values)
    if not len(values):
        return {k: None for k in ("min", "median", "p90", "p99", "max", "mean")}
    q = np.quantile(values, [0, 0.5, 0.9, 0.99, 1])
    return dict(
        zip(["min", "median", "p90", "p99", "max"], map(float, q), strict=True)
    ) | {"mean": float(np.mean(values))}


def _hist_quantiles(histogram):
    n = int(histogram.sum())
    if not n:
        return {k: None for k in ("min", "median", "p90", "p99", "max", "mean")}
    cumulative = histogram.cumsum()
    result = {}
    for name, q in zip(
        ["min", "median", "p90", "p99", "max"], [0, 0.5, 0.9, 0.99, 1], strict=True
    ):
        rank = (n - 1) * q
        low, high = int(np.floor(rank)), int(np.ceil(rank))
        left, right = np.searchsorted(cumulative, [low + 1, high + 1])
        result[name] = float(left + (right - left) * (rank - low))
    result["mean"] = float(
        np.dot(np.arange(len(histogram), dtype=np.float64), histogram) / n
    )
    return result


def _region_state(bounds):
    columns = int(np.ceil(bounds[2] - bounds[0]))
    rows = int(np.ceil(bounds[3] - bounds[1]))
    if columns * rows > 2_000_000:
        raise ProtocolViolation("Density raster exceeds bounded allocation")
    return {
        "bounds": bounds,
        "count": 0,
        "minimum": np.full(3, np.inf),
        "maximum": np.full(3, -np.inf),
        "density": np.zeros((rows, columns), np.int64),
        "histograms": np.zeros((3, 65536), np.int64),
        "rgb_any_zero": 0,
        "rgb_all_zero": 0,
        "rgb_any_storage_max": 0,
        "rgb_all_storage_max": 0,
        "rgb_all_le255": 0,
        "hag": [],
        "dtm_at_points_finite": 0,
    }


def _update_region(state, xyz, rgb, hag, dtm_available):
    bounds = state["bounds"]
    mask = (
        (xyz[:, 0] >= bounds[0])
        & (xyz[:, 0] < bounds[2])
        & (xyz[:, 1] >= bounds[1])
        & (xyz[:, 1] < bounds[3])
    )
    xyz, rgb, hag, available = xyz[mask], rgb[mask], hag[mask], dtm_available[mask]
    n = len(xyz)
    if not n:
        return
    state["count"] += n
    state["minimum"] = np.minimum(state["minimum"], xyz.min(axis=0))
    state["maximum"] = np.maximum(state["maximum"], xyz.max(axis=0))
    column = np.floor(xyz[:, 0] - bounds[0]).astype(int)
    row = np.floor(xyz[:, 1] - bounds[1]).astype(int)
    np.add.at(state["density"], (row, column), 1)
    for channel in range(3):
        state["histograms"][channel] += np.bincount(rgb[:, channel], minlength=65536)
    state["rgb_any_zero"] += int((rgb == 0).any(axis=1).sum())
    state["rgb_all_zero"] += int((rgb == 0).all(axis=1).sum())
    state["rgb_any_storage_max"] += int((rgb == 65535).any(axis=1).sum())
    state["rgb_all_storage_max"] += int((rgb == 65535).all(axis=1).sum())
    state["rgb_all_le255"] += int((rgb <= 255).all(axis=1).sum())
    state["hag"].append(hag.copy())
    state["dtm_at_points_finite"] += int(available.sum())


def _finish_region(state):
    n = state["count"]
    if not n:
        raise ProtocolViolation("No points available for a declared development region")
    bounds = state["bounds"]
    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    density = state["density"].ravel()
    occupied = density > 0
    hag = np.concatenate(state["hag"])
    finite_hag = np.isfinite(hag)
    channels = {}
    values = np.arange(65536)
    for name, histogram in zip(
        ["red", "green", "blue"], state["histograms"], strict=True
    ):
        channels[name] = {
            "stored_integer_statistics": _hist_quantiles(histogram),
            "distinct_stored_values": int(np.count_nonzero(histogram)),
            "zero_count": int(histogram[0]),
            "zero_rate": float(histogram[0] / n),
            "value_255_count": int(histogram[255]),
            "value_255_rate": float(histogram[255] / n),
            "storage_max_65535_count": int(histogram[65535]),
            "storage_max_65535_rate": float(histogram[65535] / n),
            "fraction_values_divisible_by_256": float(
                histogram[values % 256 == 0].sum() / n
            ),
            "fraction_values_divisible_by_257": float(
                histogram[values % 257 == 0].sum() / n
            ),
            "divisibility_interpretation": "descriptive stored-value pattern only; source bit depth/encoding not established",
        }
    rgb_counts = {
        name: state[key]
        for name, key in [
            ("any_channel_zero", "rgb_any_zero"),
            ("all_channels_zero", "rgb_all_zero"),
            ("any_channel_at_storage_max_65535", "rgb_any_storage_max"),
            ("all_channels_at_storage_max_65535", "rgb_all_storage_max"),
            ("all_channels_at_most_255", "rgb_all_le255"),
        ]
    }
    return {
        "point_count": n,
        "fixed_xy_bounds": bounds,
        "observed_xyz_min": state["minimum"].tolist(),
        "observed_xyz_max": state["maximum"].tolist(),
        "observed_xyz_extent_ft": (state["maximum"] - state["minimum"]).tolist(),
        "fixed_rectangle_area_sqft": area,
        "all_points_per_rectangle_sqft": n / area,
        "density_1ft_xy_cells": {
            "origin": bounds[:2],
            "cell_size_ft": 1.0,
            "shape_rows_columns": list(state["density"].shape),
            "total_cells": len(density),
            "occupied_cells": int(occupied.sum()),
            "empty_cells": int((~occupied).sum()),
            "occupancy_fraction": float(occupied.mean()),
            "counts_in_all_cells": _quantiles(density),
            "counts_in_occupied_cells": _quantiles(density[occupied]),
            "interpretation": "counts include all heights/overlapping surfaces; not independent spatial observations or a surface sampling rate",
        },
        "rgb": {
            "storage_type": "unsigned 16-bit channels",
            "channels": channels,
            "point_level_counts": rgb_counts,
            "point_level_rates": {k: v / n for k, v in rgb_counts.items()},
            "endpoint_caveat": "65535 is the unsigned 16-bit storage endpoint; 255 is not treated as saturation. Physical optical saturation, original radiometric bit depth, gamma, encoding and color registration remain UNKNOWN - needs provider/PI.",
        },
        "hag_ft": {
            "point_count": n,
            "finite_count": int(finite_hag.sum()),
            "nonfinite_count": int((~finite_hag).sum()),
            "finite_statistics": _quantiles(hag[finite_hag]),
            "negative_count": int((finite_hag & (hag < 0)).sum()),
            "negative_fraction_all_points": float((finite_hag & (hag < 0)).mean()),
            "zero_count": int((finite_hag & (hag == 0)).sum()),
            "interpretation": "z minus generated terrain estimate; negative values are observed residuals, not a direct accuracy estimate",
        },
        "dtm_at_point_locations": {
            "finite_count": state["dtm_at_points_finite"],
            "finite_fraction": state["dtm_at_points_finite"] / n,
            "interpretation": "finite coverage of generated DTM; may include interpolated/global-minimum-filled cells and does not establish measured-ground support",
        },
    }


def collect_quality(run_dir, repository):
    repository = Path(repository).resolve()
    run_dir = Path(run_dir).resolve()
    try:
        run_dir.relative_to(repository / "revision_work/runs")
    except ValueError as exc:
        raise ProtocolViolation(
            "Data-quality input must be a bounded revision run"
        ) from exc
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"]["input"] != "data/eval/tile_c.las":
        raise ProtocolViolation("Descriptive audit is restricted to raw tile C")
    guard = HoldoutGuard.from_files(
        repository / "holdout.json",
        repository / "objective.md",
        expected_holdout_sha256=manifest["protocol"]["holdout_sha256"],
        expected_objective_sha256=manifest["protocol"]["objective_sha256"],
    )
    protocol = guard._protocol()
    context = list(
        guard.assert_development_bounds(manifest["dataset"]["context_bounds"])
    )
    pilot = list(guard.assert_development_bounds(manifest["dataset"]["score_bounds"]))
    if pilot != protocol["development"]["scoring_bounds"]:
        raise ProtocolViolation("Pilot bounds differ from preregistered region")
    input_path = run_dir / "input.las"
    preprocessing = json.loads((run_dir / "preprocessing.json").read_text())
    input_hash = sha256_file(input_path)
    if input_hash != preprocessing["input_crop_sha256"]:
        raise ProtocolViolation(
            "Bounded input differs from recorded preprocessing hash"
        )
    with np.load(run_dir / "slices/grid_meta.npz") as saved:
        grid = {k: saved[k].item() for k in saved.files}
    hag = np.load(run_dir / "derived/hag.npy", mmap_mode="r")
    dtm = np.load(run_dir / "derived/dtm.npy", mmap_mode="r")
    if dtm.shape != (int(grid["rows"]), int(grid["cols"])):
        raise ProtocolViolation("DTM shape does not match bounded run grid")
    states = {"context_c": _region_state(context), "pilot_c": _region_state(pilot)}
    grid_occupancy = np.zeros(dtm.shape, dtype=bool)
    offset = 0
    with laspy.open(input_path) as reader:
        if hag.shape != (reader.header.point_count,):
            raise ProtocolViolation("HAG array does not match input record count")
        units = validate_coordinate_units(reader.header, assumed_units="US survey foot")
        for chunk in reader.chunk_iterator(200_000):
            n = len(chunk)
            xyz = np.column_stack([chunk.x, chunk.y, chunk.z])
            if not np.isfinite(xyz).all():
                raise ProtocolViolation("Nonfinite bounded input coordinates")
            inside = (
                (xyz[:, 0] >= context[0])
                & (xyz[:, 0] < context[2])
                & (xyz[:, 1] >= context[1])
                & (xyz[:, 1] < context[3])
            )
            if not inside.all():
                raise ProtocolViolation(
                    "Bounded input contains points outside allowed development context"
                )
            rgb = np.column_stack([chunk.red, chunk.green, chunk.blue])
            rows = np.clip(
                ((grid["y_max"] - xyz[:, 1]) / grid["resolution"]).astype(int),
                0,
                dtm.shape[0] - 1,
            )
            columns = np.clip(
                ((xyz[:, 0] - grid["x_min"]) / grid["resolution"]).astype(int),
                0,
                dtm.shape[1] - 1,
            )
            grid_occupancy[rows, columns] = True
            dtm_available = np.isfinite(dtm[rows, columns])
            chunk_hag = np.asarray(hag[offset : offset + n])
            for state in states.values():
                _update_region(state, xyz, rgb, chunk_hag, dtm_available)
            offset += n
    if offset != len(hag):
        raise ProtocolViolation("Input record count changed while reading")
    if sha256_file(input_path) != input_hash:
        raise ProtocolViolation(
            "Bounded input changed while generating descriptive report"
        )
    finite_dtm = np.isfinite(dtm)
    sources = [
        "manifest.json",
        "preprocessing.json",
        "input.las",
        "derived/hag.npy",
        "derived/dtm.npy",
        "slices/grid_meta.npz",
    ]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "M5 descriptive quality evidence on fixed development data; no experiment selection",
        "source_run": str(run_dir.relative_to(repository)),
        "source_hashes": {p: sha256_file(run_dir / p) for p in sources},
        "script_sha256": sha256_file(__file__),
        "units_validator_sha256": sha256_file(Path(__file__).with_name("units.py")),
        "protocol_hashes": guard.verify_locked_files(),
        "units": units,
        "sensor": UNKNOWN,
        "acquisition_date": UNKNOWN,
        "colorization": UNKNOWN,
        "color_registration_accuracy": UNKNOWN,
        "radiometric_encoding": UNKNOWN,
        "reference_or_prediction_data_read": False,
        "regions": {name: _finish_region(state) for name, state in states.items()},
        "dtm_grid": {
            "shape": list(dtm.shape),
            "resolution_ft": grid["resolution"],
            "finite_cells": int(finite_dtm.sum()),
            "nonfinite_cells": int((~finite_dtm).sum()),
            "finite_fraction": float(finite_dtm.mean()),
            "finite_elevation_ft": _quantiles(dtm[finite_dtm]),
            "input_occupied_cells": int(grid_occupancy.sum()),
            "input_empty_cells": int((~grid_occupancy).sum()),
            "finite_dtm_cells_without_input_points": int(
                (finite_dtm & ~grid_occupancy).sum()
            ),
            "observed_ground_support_mask": "NOT SAVED by Phase 1 pipeline; cannot distinguish observed-ground/interpolated/fallback-filled terrain cells from final DTM alone",
            "accuracy": "Not evaluated: no independent terrain checkpoints or ground reference; finite support is not vertical accuracy",
        },
        "limitations": [
            "One fixed development context and one nested pilot, not independent sites.",
            "High XY point counts include multiple heights, vertical surfaces and potential acquisition overlap; points are not independent observations.",
            "Observed RGB storage endpoints and code patterns do not establish optical saturation, bit-depth provenance, colorization method or registration accuracy.",
            "Negative HAG and DTM coverage are descriptive residual/support checks; no absolute or relative terrain accuracy is claimed.",
            "Acquisition and color-registration provenance require provider/PI confirmation.",
        ],
    }


def render_markdown(report):
    lines = [
        "# M5 bounded development-data quality report",
        "",
        f"Generated {report['created_utc']}. Source: `{report['source_run']}/input.las`.",
        "",
        "This is descriptive evidence on fixed development data. No reference classes, model predictions, A/B inputs or full-survey inputs were read. It was not used to select an experiment.",
        "",
        (
            "Sensor, acquisition date, colorization, radiometric encoding and color-registration accuracy: **UNKNOWN - needs provider/PI**. "
            f"CRS: {report['units']['crs']}. Horizontal unit status: {report['units']['horizontal_units_status']}; "
            f"vertical unit status: {report['units']['vertical_units_status']}. Vertical datum: {report['units']['vertical_datum']}. "
            "The configured distance unit is US survey foot (1200/3937 metre)."
        ),
        "",
        "## Bounded context and nested pilot",
        "",
        "| Measure | Tile-C context | Pilot |",
        "|---|---:|---:|",
    ]
    context, pilot = [report["regions"][k] for k in ["context_c", "pilot_c"]]

    def row(name, values):
        lines.append(f"| {name} | {values[0]} | {values[1]} |")

    row("Points", [f"{r['point_count']:,}" for r in [context, pilot]])
    row(
        "Fixed XY rectangle area (ft²)",
        [f"{r['fixed_rectangle_area_sqft']:,.1f}" for r in [context, pilot]],
    )
    row(
        "Points per fixed rectangle ft²",
        [f"{r['all_points_per_rectangle_sqft']:.3f}" for r in [context, pilot]],
    )
    row(
        "Observed XYZ extent (ft)",
        [
            ", ".join(f"{v:.3f}" for v in r["observed_xyz_extent_ft"])
            for r in [context, pilot]
        ],
    )
    row(
        "Occupied 1 ft XY cells / total",
        [
            f"{r['density_1ft_xy_cells']['occupied_cells']:,} / {r['density_1ft_xy_cells']['total_cells']:,}"
            for r in [context, pilot]
        ],
    )
    for key, label in [
        ("min", "Minimum"),
        ("median", "Median"),
        ("p90", "90th percentile"),
        ("p99", "99th percentile"),
        ("max", "Maximum"),
    ]:
        row(
            f"{label} points per 1 ft cell (including empty cells)",
            [
                f"{r['density_1ft_xy_cells']['counts_in_all_cells'][key]:.2f}"
                for r in [context, pilot]
            ],
        )
    row(
        "Any RGB channel zero",
        [
            f"{r['rgb']['point_level_counts']['any_channel_zero']:,} ({r['rgb']['point_level_rates']['any_channel_zero']:.4%})"
            for r in [context, pilot]
        ],
    )
    row(
        "All RGB channels zero",
        [
            f"{r['rgb']['point_level_counts']['all_channels_zero']:,} ({r['rgb']['point_level_rates']['all_channels_zero']:.4%})"
            for r in [context, pilot]
        ],
    )
    row(
        "Any RGB channel at 65535",
        [
            f"{r['rgb']['point_level_counts']['any_channel_at_storage_max_65535']:,} ({r['rgb']['point_level_rates']['any_channel_at_storage_max_65535']:.4%})"
            for r in [context, pilot]
        ],
    )
    row(
        "Negative HAG points",
        [
            f"{r['hag_ft']['negative_count']:,} ({r['hag_ft']['negative_fraction_all_points']:.4%})"
            for r in [context, pilot]
        ],
    )
    row(
        "HAG minimum / maximum (ft)",
        [
            f"{r['hag_ft']['finite_statistics']['min']:.5f} / {r['hag_ft']['finite_statistics']['max']:.5f}"
            for r in [context, pilot]
        ],
    )
    lines.extend(
        [
            "",
            "Density includes all heights and possible multiple surfaces/overlap in each XY cell. These counts are not independent observations or a terrain-surface sampling rate. Fixed rectangle area and actual observed XYZ bounds are separate fields in the JSON.",
            "",
            "## Stored RGB values",
            "",
            "| Region | Channel | Min | Median | p90 | p99 | Max | Values equal to 255 | Values equal to 65535 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for region_name, region in report["regions"].items():
        for channel, data in region["rgb"]["channels"].items():
            stats = data["stored_integer_statistics"]
            lines.append(
                f"| {region_name} | {channel} | "
                + " | ".join(
                    f"{stats[k]:.1f}" for k in ["min", "median", "p90", "p99", "max"]
                )
                + f" | {data['value_255_count']:,} | {data['storage_max_65535_count']:,} |"
            )
    lines.extend(
        [
            "",
            "The endpoint 65535 is the unsigned 16-bit storage limit; **255 is not called saturation**. Storage endpoint counts do not establish optical saturation or original radiometric bit depth. The JSON includes exact zero/endpoint rates, distinct values, and divisibility patterns as observations only. No provider encoding or color-registration quality is inferred.",
            "",
            "## Generated terrain support",
            "",
        ]
    )
    dtm = report["dtm_grid"]
    lines.append(
        f"The final {dtm['shape'][0]} × {dtm['shape'][1]} DTM at {dtm['resolution_ft']} ft resolution has {dtm['finite_cells']:,} finite cells ({dtm['finite_fraction']:.4%}); {dtm['finite_dtm_cells_without_input_points']:,} finite cells contain no input point in their XY cell. All-height input occupancy is not observed-ground support."
    )
    lines.extend(
        [
            "",
            f"**{dtm['observed_ground_support_mask']}.** No independent terrain checkpoints or ground reference were used. Negative HAG is a measured residual against this estimated DTM, not proof of an acquisition error or a vertical-accuracy metric.",
            "",
            "## Provenance and unresolved work",
            "",
            f"Input crop SHA-256: `{report['source_hashes']['input.las']}`. Script SHA-256: `{report['script_sha256']}`. The JSON records all input/artifact hashes and exact statistics.",
            "",
        ]
    )
    lines.extend([f"- {text}" for text in report["limitations"]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()
    stem = Path(args.output_stem)
    paths = [stem.with_suffix(".json"), stem.with_suffix(".md")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite a previous data-quality report")
    report = collect_quality(args.run_dir, args.repository)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with paths[1].open("x") as stream:
        stream.write(render_markdown(report))
    print(
        json.dumps(
            {
                "outputs": [str(p) for p in paths],
                "counts": {k: v["point_count"] for k, v in report["regions"].items()},
                "input_sha256": report["source_hashes"]["input.las"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
