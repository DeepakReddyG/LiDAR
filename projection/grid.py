"""Initialize the shared pixel/world grid from LAS header bounds, without
reading point records. Run before ground/ortho via `main.py --stage grid`.

The origin is (x_min, y_max); rows increase southward. Dimensions use ceiling
division, with at least one cell per axis. Consumers clip points exactly on
the far boundary into the last cell, preserving the existing grid convention.
Matching metadata is left untouched. A conflicting grid is rejected so cached
rasters and masks cannot silently acquire a different coordinate mapping.
"""

import argparse
import math
from pathlib import Path

import laspy
import numpy as np

from config import GRID_META_PATH, GRID_RESOLUTION, LAS_PATH


def grid_from_header(header, resolution: float = GRID_RESOLUTION) -> dict:
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Grid resolution must be finite and positive")
    if header.point_count == 0:
        raise ValueError("Cannot define a grid for an empty LAS survey")
    x_min, y_min = map(float, header.mins[:2])
    x_max, y_max = map(float, header.maxs[:2])
    if not all(math.isfinite(v) for v in (x_min, y_min, x_max, y_max)):
        raise ValueError("LAS XY bounds must be finite")
    if x_max < x_min or y_max < y_min:
        raise ValueError("LAS XY bounds are reversed")
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "resolution": resolution,
        "rows": max(1, math.ceil((y_max - y_min) / resolution)),
        "cols": max(1, math.ceil((x_max - x_min) / resolution)),
    }


def run_grid(
    las_path=LAS_PATH, out_path=GRID_META_PATH, resolution: float = GRID_RESOLUTION
) -> None:
    with laspy.open(las_path) as reader:
        grid = grid_from_header(reader.header, resolution)
    out_path = Path(out_path)
    if out_path.exists():
        with np.load(out_path) as saved:
            matches = all(
                k in saved and np.array_equal(saved[k], v) for k, v in grid.items()
            )
        if not matches:
            raise ValueError(
                f"Existing grid {out_path} conflicts with LAS bounds or resolution. "
                "Preserve existing outputs and use a separate data directory for "
                "a different survey/grid; dependent rasters and masks need rebuilding."
            )
        print(f"Grid already matches: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **grid)
    print(f"Grid {grid['rows']} × {grid['cols']} at {resolution} ft/px → {out_path}")


def _self_check() -> None:
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.point_count = 2
    header.mins = np.array([10.0, 20.0, 0.0])
    header.maxs = np.array([12.25, 23.0, 0.0])
    grid = grid_from_header(header, 0.5)
    assert (grid["rows"], grid["cols"]) == (6, 5)
    header.maxs = header.mins.copy()
    grid = grid_from_header(header, 0.5)
    assert (grid["rows"], grid["cols"]) == (1, 1)
    print("self-check OK: non-square grid and single-location bounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    if parser.parse_args().self_check:
        _self_check()
    else:
        run_grid()
