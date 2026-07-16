import laspy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_point_cloud(las_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a .las file and return XY coordinates and Z values.
    Returns:
        xy: (N, 2) array of X, Y coordinates
        z:  (N,)  array of Z (elevation) values
    """
    las = laspy.read(las_path)
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    print(f"Loaded {len(x):,} points from {Path(las_path).name}")
    print(f"  X range: {x.min():.2f} → {x.max():.2f}")
    print(f"  Y range: {y.min():.2f} → {y.max():.2f}")
    print(f"  Z range: {z.min():.2f} → {z.max():.2f}")

    xy = np.stack([x, y], axis=1)
    return xy, z


def _uniform_filter_2d(arr: np.ndarray, size: int) -> np.ndarray:
    """
    Fast 2D box-filter (uniform mean) via a 1-indexed Summed Area Table.
    Equivalent to scipy.ndimage.uniform_filter but numpy-only.

    Padding: size//2 before, size-size//2-1 after → total pad = size-1.
    The 1-indexed SAT avoids all off-by-one issues with any window size.
    """
    rows, cols = arr.shape
    h_lo = size // 2
    h_hi = size - h_lo - 1          # asymmetric for even sizes; same as h_lo for odd

    padded = np.pad(arr, ((h_lo, h_hi), (h_lo, h_hi)), mode="edge")
    # padded.shape = (rows + size - 1, cols + size - 1)

    # Build 1-indexed SAT: cs[i, j] = sum of padded[:i, :j]
    cs = np.zeros((rows + size, cols + size), dtype=np.float64)
    cs[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)

    # For pixel (r, c) the window in padded covers rows [r, r+size), cols [c, c+size)
    # In 1-indexed SAT: sum = cs[r+size, c+size] - cs[r, c+size] - cs[r+size, c] + cs[r, c]
    out = (
        cs[size:size + rows, size:size + cols]
        - cs[:rows,          size:size + cols]
        - cs[size:size + rows, :cols]
        + cs[:rows,            :cols]
    ) / (size * size)

    return out.astype(arr.dtype)


def _stretch(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Percentile-stretch arr to [0, 1]."""
    lo, hi = np.nanpercentile(arr, p_lo), np.nanpercentile(arr, p_hi)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def project_top_down(
    xy: np.ndarray,
    z: np.ndarray,
    resolution: float = 0.5,
    output_path: str = "data/slices/top_down.png",
) -> tuple[np.ndarray, dict]:
    """
    Project 3D point cloud to a 2D top-down image and save as a
    3-channel false-colour RGB PNG optimised for SAM2 segmentation:

        R — global elevation (absolute height contrast)
        G — local relief model (LRM): elevation minus local mean
            highlights fine surface features above the ground plane
        B — slope (gradient magnitude): bright at edges / structure boundaries

    Also saves:
        data/slices/elevation_grid.npy   — raw max-Z grid (float32)
        data/slices/grid_meta.npz        — coordinate metadata for map_back.py

    Returns:
        grid:  (H, W) float32 elevation grid (NaN where no points)
        meta:  dict with x_min, y_min, resolution, rows, cols
    """
    x, y = xy[:, 0], xy[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    cols = int(np.ceil((x_max - x_min) / resolution))
    rows = int(np.ceil((y_max - y_min) / resolution))
    print(f"\nGrid size: {rows} rows × {cols} cols at {resolution} ft/pixel")

    # ── rasterise: max-Z per cell ──────────────────────────────────────────
    col_idx = np.clip(((x - x_min) / resolution).astype(np.int32), 0, cols - 1)
    row_idx = np.clip(((y_max - y) / resolution).astype(np.int32), 0, rows - 1)

    grid = np.full((rows, cols), np.nan, dtype=np.float32)
    # np.fmax ignores NaN (unlike np.maximum which propagates it),
    # so cells initialised to NaN get overwritten by the first point that lands there.
    np.fmax.at(grid, (row_idx, col_idx), z.astype(np.float32))

    # ── fill NaN cells with local box-filter mean, then global min ─────────
    grid_filled = grid.copy()
    nan_mask = np.isnan(grid_filled)
    if nan_mask.any():
        g_min = float(np.nanmin(grid))   # safe: at least some cells are filled
        grid_tmp = np.where(nan_mask, g_min, grid)
        local_fill = _uniform_filter_2d(grid_tmp, size=9)
        grid_filled[nan_mask] = local_fill[nan_mask]
        # Any remaining NaN (edge case: entirely empty neighbourhood) → global min
        grid_filled[np.isnan(grid_filled)] = g_min

    # ── Channel R: global elevation ─────────────────────────────────────────
    r = _stretch(grid_filled, p_lo=2, p_hi=98)

    # ── Channel G: local relief model (LRM) ────────────────────────────────
    # Subtract a wide local mean (~25 ft window at 0.5 ft/px = 51 px)
    window = max(3, int(25.0 / resolution) | 1)   # ensure odd
    local_mean = _uniform_filter_2d(grid_filled, size=window)
    lrm = grid_filled - local_mean
    g = _stretch(lrm, p_lo=5, p_hi=95)

    # ── Channel B: slope (gradient magnitude) ───────────────────────────────
    gy, gx = np.gradient(grid_filled)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    b = _stretch(slope, p_lo=2, p_hi=98)

    # ── Stack → uint8 RGB ───────────────────────────────────────────────────
    rgb = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(output_path)
    print(f"Saved false-colour slice → {output_path}")
    print(f"  R=elevation  G=local-relief  B=slope")

    # ── Save elevation grid for classify.py ─────────────────────────────────
    grid_path = out_dir / "elevation_grid.npy"
    np.save(grid_path, grid_filled)
    print(f"Saved elevation grid (NaN-filled) → {grid_path}")

    # Save the pre-fill grid so classify.py can derive the exact no-data mask
    raw_grid_path = out_dir / "elevation_grid_raw.npy"
    np.save(raw_grid_path, grid)   # NaN where no LiDAR point landed
    print(f"Saved elevation grid (raw, NaN=no-data) → {raw_grid_path}")

    # ── Save coordinate metadata for map_back.py ────────────────────────────
    meta = dict(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                resolution=resolution, rows=rows, cols=cols)
    meta_path = out_dir / "grid_meta.npz"
    np.savez(meta_path, **meta)
    print(f"Saved grid metadata → {meta_path}")

    return grid, meta


if __name__ == "__main__":
    LAS_PATH = "data/raw/UPark_Merged_PS_NAD83_G18_USFT_las.las"
    OUTPUT_PATH = "data/slices/top_down.png"
    RESOLUTION = 0.5  # 0.5 ft per pixel

    xy, z = load_point_cloud(LAS_PATH)
    grid, meta = project_top_down(xy, z, resolution=RESOLUTION, output_path=OUTPUT_PATH)

    print("\nDone! Open data/slices/top_down.png to see the false-colour slice.")
    print("  Red channel  — elevation (absolute height)")
    print("  Green channel — local relief (surface features)")
    print("  Blue channel  — slope (edges / structure boundaries)")
