"""evaluation/crop_tiles.py — chunked XY-crop of 3 ground-truth tiles from the
raw LAS (MANUAL §6.0). Never loads the full 411 M-point file: streams
CHUNK_SIZE points at a time and keeps only the tiny fraction inside each
tile's bounds.

Stamps each surviving point's original global index as an extra `orig_index`
uint32 dimension — point order does not survive a CloudCompare edit, so all
later GT/prediction matching (evaluate.py) uses this stamp, never file order.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from config import CHUNK_SIZE, EVAL_DIR, EVAL_TILE_BOUNDS, LAS_PATH


def crop_tile(
    las_path: str | Path,
    bounds: tuple[float, float, float, float],
    out_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Crop points within `bounds` (x0, y0, x1, y1) into `out_path`, stamping
    each point's original index. Returns the number of points written."""
    x0, y0, x1, y1 = bounds

    with laspy.open(las_path) as reader:
        dims = [d.name for d in reader.header.point_format.dimensions]
        collected: dict[str, list[np.ndarray]] = {d: [] for d in dims}
        orig_index_parts: list[np.ndarray] = []

        offset = 0
        for chunk in reader.chunk_iterator(chunk_size):
            n = len(chunk)
            mask = (chunk.x >= x0) & (chunk.x <= x1) & (chunk.y >= y0) & (chunk.y <= y1)
            if mask.any():
                orig_index_parts.append(
                    np.arange(offset, offset + n, dtype=np.uint32)[mask]
                )
                for d in dims:
                    collected[d].append(np.asarray(getattr(chunk, d))[mask])
            offset += n

        n_out = sum(len(p) for p in orig_index_parts)
        if n_out == 0:
            raise ValueError(f"No points found in bounds {bounds} for {out_path}")

        out_header = laspy.LasHeader(
            point_format=reader.header.point_format,
            version=reader.header.version,
        )
        out_header.scales = reader.header.scales
        out_header.offsets = reader.header.offsets
        out_header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="orig_index",
                type=np.uint32,
                description="index into source LAS",
            )
        )

        out_las = laspy.LasData(out_header)
        for d in dims:
            setattr(out_las, d, np.concatenate(collected[d]))
        out_las.orig_index = np.concatenate(orig_index_parts)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_las.write(str(out_path))

    print(f"  {out_path}: {n_out:,} points")
    return n_out


def main() -> None:
    for name, bounds in EVAL_TILE_BOUNDS.items():
        out_path = EVAL_DIR / f"tile_{name}.las"
        crop_tile(LAS_PATH, bounds, out_path)

    print(
        "\n"
        "MANUAL STEP — hand-label these tiles in CloudCompare:\n"
        "  1. Create a safe copy with evaluation.annotation prepare.\n"
        "     Keep original tiles outside CloudCompare; large orig_index\n"
        "     values are rounded on export. Copies over 2^24 points need subdivision.\n"
        "  2. Segment class parts and export LAS with tile_index retained.\n"
        "  3. Recover exact orig_index with evaluation.annotation restore.\n"
        "  4. Assemble restored parts with evaluation.merge_gt_parts.\n"
        "     See MANUAL §6.0 for commands, class rules, and validation.\n"
    )


if __name__ == "__main__":
    main()
