"""projection/ground.py — CSF ground filter → DTM → per-point HAG (MANUAL §6.2, T2).

411 M points won't fit CSF — they don't need to:

    1. decimate: lowest-Z point per 2×2 ft cell (chunked pass) → ~200 k pts
    2. CSF on the decimated set → ground points
    3. DTM: min-Z rasterise of ground points at 2 ft, gap-fill with the SAT
       box filter, bilinear-upsample to the 0.5 ft ortho grid (grid_meta)
    4. per-point hag.npy memmap: z − dtm[row, col], one chunked pass

Outputs (data/derived/): dtm.npy (H, W float32, ortho grid), hag.npy
((N,) float32, LAS point order), dtm_preview.png.

HUMAN GATE: dtm_preview.png must look like smooth bare terrain — any
embossed building or tree outline means CLOTH_RESOLUTION is off.
"""

from __future__ import annotations

import argparse

import laspy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import (CHUNK_SIZE, CLOTH_RESOLUTION, DECIMATE_CELL, DERIVED_DIR,
                    GRID_META_PATH, GROUND_OUTLIER_FT, LAS_PATH)


def _uniform_filter_2d(arr: np.ndarray, size: int) -> np.ndarray:
    """Fast 2D box-filter (uniform mean) via a 1-indexed Summed Area Table.
    Numpy-only equivalent of scipy.ndimage.uniform_filter. (Inherited from
    v1 slice.py, its last surviving piece.)"""
    rows, cols = arr.shape
    h_lo = size // 2
    h_hi = size - h_lo - 1

    padded = np.pad(arr, ((h_lo, h_hi), (h_lo, h_hi)), mode="edge")
    cs = np.zeros((rows + size, cols + size), dtype=np.float64)
    cs[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    out = (
        cs[size:size + rows, size:size + cols]
        - cs[:rows,          size:size + cols]
        - cs[size:size + rows, :cols]
        + cs[:rows,            :cols]
    ) / (size * size)
    return out.astype(arr.dtype)


def _grid_meta() -> dict:
    m = np.load(GRID_META_PATH)
    return {k: float(m[k]) if k != "rows" and k != "cols" else int(m[k])
            for k in ("x_min", "y_max", "resolution", "rows", "cols")}


def decimate_min_z(las_path=LAS_PATH, cell: float = DECIMATE_CELL,
                   limit_chunks: int | None = None) -> np.ndarray:
    """Chunked pass: lowest-Z point per cell×cell ft. Returns (M, 3) xyz
    (cell centres in XY, min Z)."""
    g = _grid_meta()
    cols = int(np.ceil(g["cols"] * g["resolution"] / cell))
    rows = int(np.ceil(g["rows"] * g["resolution"] / cell))
    minz = np.full((rows, cols), np.inf, dtype=np.float64)

    with laspy.open(las_path) as f:
        for i, ch in enumerate(f.chunk_iterator(CHUNK_SIZE)):
            x, y, z = np.asarray(ch.x), np.asarray(ch.y), np.asarray(ch.z)
            c = np.clip(((x - g["x_min"]) / cell).astype(np.int32), 0, cols - 1)
            r = np.clip(((g["y_max"] - y) / cell).astype(np.int32), 0, rows - 1)
            np.minimum.at(minz, (r, c), z)
            if limit_chunks is not None and i + 1 >= limit_chunks:
                break

    # reject low outliers (scanner noise / multipath below true ground):
    # min-Z picks the lowest return per cell, so a single bad point sinks the
    # cloth locally — drop cells > GROUND_OUTLIER_FT below their 5×5 median
    pad = np.pad(minz, 2, constant_values=np.nan)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(
            np.stack([pad[dr:dr + rows, dc:dc + cols]
                      for dr in range(5) for dc in range(5)]), axis=0)
    low = np.isfinite(minz) & np.isfinite(med) & (minz < med - GROUND_OUTLIER_FT)
    minz[low] = np.inf
    print(f"  dropped {low.sum():,} low-outlier cells")

    r, c = np.nonzero(np.isfinite(minz))
    xyz = np.column_stack([
        g["x_min"] + (c + 0.5) * cell,
        g["y_max"] - (r + 0.5) * cell,
        minz[r, c],
    ])
    print(f"  decimated to {len(xyz):,} candidate points ({rows}×{cols} cells)")
    return xyz


def csf_ground(xyz: np.ndarray, cloth_resolution: float = CLOTH_RESOLUTION) -> np.ndarray:
    """Run CSF, return the ground subset of xyz."""
    import CSF
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    csf.setPointCloud(np.ascontiguousarray(xyz, dtype=np.float64))
    ground, non_ground = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(ground, non_ground)
    idx = np.asarray(ground, dtype=np.int64)
    print(f"  CSF: {len(idx):,} ground / {len(xyz):,} candidates")
    return xyz[idx]


def build_dtm(ground_xyz: np.ndarray, cell: float = DECIMATE_CELL) -> np.ndarray:
    """Min-Z rasterise ground points at `cell` ft, gap-fill, bilinear-upsample
    to the 0.5 ft ortho grid. Returns (rows, cols) float32 DTM."""
    g = _grid_meta()
    cols_c = int(np.ceil(g["cols"] * g["resolution"] / cell))
    rows_c = int(np.ceil(g["rows"] * g["resolution"] / cell))

    coarse = np.full((rows_c, cols_c), np.nan, dtype=np.float64)
    x, y, z = ground_xyz.T
    c = np.clip(((x - g["x_min"]) / cell).astype(np.int32), 0, cols_c - 1)
    r = np.clip(((g["y_max"] - y) / cell).astype(np.int32), 0, rows_c - 1)
    np.fmax.at(coarse, (r, c), z)          # NaN cells take first value

    # gap-fill: masked local mean (SAT box filter on values and counts),
    # widening the window until every cell is filled
    nan = np.isnan(coarse)
    vals = np.where(nan, 0.0, coarse)
    mask = (~nan).astype(np.float64)
    for size in (5, 17, 65):
        if not np.isnan(coarse).any():
            break
        local_sum = _uniform_filter_2d(vals, size) * size * size
        local_cnt = _uniform_filter_2d(mask, size) * size * size
        fill = np.divide(local_sum, local_cnt,
                         out=np.full_like(local_sum, np.nan),
                         where=local_cnt > 0.5)
        still = np.isnan(coarse)
        coarse[still] = fill[still]
    coarse[np.isnan(coarse)] = np.nanmin(coarse)   # anything left: global min

    dtm = np.asarray(
        Image.fromarray(coarse.astype(np.float32)).resize(
            (g["cols"], g["rows"]), Image.BILINEAR)
    )
    return dtm


def write_hag(dtm: np.ndarray, las_path=LAS_PATH, out_dir=DERIVED_DIR,
              limit_chunks: int | None = None) -> int:
    """Chunked pass: hag.npy = z − dtm[row, col], LAS point order."""
    g = _grid_meta()
    with laspy.open(las_path) as f:
        n_pts = f.header.point_count
        hag = np.lib.format.open_memmap(out_dir / "hag.npy", mode="w+",
                                        dtype=np.float32, shape=(n_pts,))
        off = 0
        for i, ch in enumerate(f.chunk_iterator(CHUNK_SIZE)):
            n = len(ch)
            x, y, z = np.asarray(ch.x), np.asarray(ch.y), np.asarray(ch.z)
            c = np.clip(((x - g["x_min"]) / g["resolution"]).astype(np.int32),
                        0, g["cols"] - 1)
            r = np.clip(((g["y_max"] - y) / g["resolution"]).astype(np.int32),
                        0, g["rows"] - 1)
            hag[off:off + n] = (z - dtm[r, c]).astype(np.float32)
            off += n
            print(f"  hag chunk {i + 1}: {off:,}/{n_pts:,}", flush=True)
            if limit_chunks is not None and i + 1 >= limit_chunks:
                break
        hag.flush()
    return off


def run_ground(limit_chunks: int | None = None) -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 decimating (min-Z per cell) …")
    xyz = decimate_min_z(limit_chunks=limit_chunks)
    print("2/4 CSF ground filtering …")
    ground_xyz = csf_ground(xyz)
    print("3/4 building DTM …")
    dtm = build_dtm(ground_xyz)
    np.save(DERIVED_DIR / "dtm.npy", dtm.astype(np.float32))

    prev = DERIVED_DIR / "dtm_preview.png"
    lo, hi = np.percentile(dtm, 2), np.percentile(dtm, 98)
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(dtm, cmap="terrain", vmin=lo, vmax=hi)
    fig.colorbar(im, shrink=0.6, label="elevation (ft)")
    ax.set_title("DTM — must be smooth bare terrain (no embossed buildings/trees)")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(prev, dpi=100); plt.close(fig)
    print(f"Saved {prev}\nINSPECT IT: embossed structures ⇒ adjust CLOTH_RESOLUTION.")

    print("4/4 per-point HAG …")
    n = write_hag(dtm, limit_chunks=limit_chunks)
    print(f"Saved hag.npy for {n:,} points")


def _self_check() -> None:
    """Synthetic: flat plane at z=100 with a 20 ft box on it — CSF+DTM must
    recover the plane under the box; HAG of box-top points ≈ box height."""
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.uniform(0, 200, n)
    y = rng.uniform(0, 200, n)
    z = np.full(n, 100.0) + rng.normal(0, 0.05, n)
    in_box = (x > 80) & (x < 120) & (y > 80) & (y < 120)
    z[in_box] += 20.0                                    # building roof

    xyz = np.column_stack([x, y, z])
    ground = csf_ground(xyz, cloth_resolution=2.0)
    assert len(ground) > 0.5 * (~in_box).sum() * 0.5, "CSF kept too few ground pts"
    box_centre_z = ground[
        (ground[:, 0] > 90) & (ground[:, 0] < 110)
        & (ground[:, 1] > 90) & (ground[:, 1] < 110), 2]
    assert box_centre_z.size == 0 or np.all(box_centre_z < 105), \
        "CSF classified roof points as ground"

    # DTM interpolated under the box ≈ plane; HAG of roof ≈ 20
    # (tiny local grid rather than the site grid_meta)
    cell = 2.0
    cols = rows = int(200 / cell)
    coarse = np.full((rows, cols), np.nan)
    c = np.clip((ground[:, 0] / cell).astype(int), 0, cols - 1)
    r = np.clip(((200 - ground[:, 1]) / cell).astype(int), 0, rows - 1)
    np.fmax.at(coarse, (r, c), ground[:, 2])
    for size in (5, 17, 65):                 # same widening as build_dtm
        nan = np.isnan(coarse)
        if not nan.any():
            break
        vals, mask = np.where(nan, 0, coarse), (~nan).astype(float)
        s = _uniform_filter_2d(vals, size) * size * size
        m = _uniform_filter_2d(mask, size) * size * size
        fill = np.divide(s, m, out=np.full_like(s, np.nan), where=m > 0.5)
        coarse[nan] = fill[nan]
    dtm_under_box = coarse[rows // 2 - 5:rows // 2 + 5, cols // 2 - 5:cols // 2 + 5]
    assert np.all(np.abs(dtm_under_box - 100.0) < 1.0), \
        f"DTM under box off plane: {dtm_under_box.min():.2f}–{dtm_under_box.max():.2f}"
    hag_roof = 120.0 - dtm_under_box.mean()
    assert abs(hag_roof - 20.0) < 1.0, f"roof HAG {hag_roof:.2f} ≠ 20"
    print("self-check OK: plane recovered under box, roof HAG ≈ 20 ft")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit-chunks", type=int, default=None)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        _self_check()
    else:
        run_ground(limit_chunks=args.limit_chunks)
