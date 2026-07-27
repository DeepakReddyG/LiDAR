"""projection/ortho.py — the raster SAM3 actually sees (MANUAL §6.3, T4).

Two chunked passes over the LAS at 0.5 ft/px (grid_meta conventions):

    pass 1  surface_z: max Z per cell (drives Z-aware map-back)
    pass 2  near-surface means: per cell, mean R,G,B / ExG / HAG / intensity
            of points within TOP_SURFACE_FT of that cell's max Z
            (top-surface colour, not colour smeared through the canopy)

Outputs (data/slices/):
    ortho_rgb.png     true-colour nadir ortho, voids inpainted (cv2 Telea —
                      photo-like texture; a box blur would leave smears no
                      photograph contains, defeating the in-distribution goal)
    surface_z.npy     (H, W) float32, NaN where void
    exg_grid.npy, hag_grid.npy, intensity_grid.npy — (H, W) float32, NaN void
    void_mask.npy     (H, W) bool — cells with zero points; labels there are
                      vetoed at fuse time
    tiles/tile_r{row0}_c{col0}.png — 1024×1024 crops, stride 768
    grid_meta.npz     x_min, y_min, x_max, y_max, resolution, rows, cols

HUMAN GATE: ortho_rgb.png must look like an aerial photo.
"""

from __future__ import annotations

import argparse

import cv2
import laspy
import numpy as np

from config import (
    CHUNK_SIZE,
    DERIVED_DIR,
    GRID_META_PATH,
    INPAINT_RADIUS_PX,
    LAS_PATH,
    RGB_16BIT_TO_8BIT_DIVISOR,
    SLICES_DIR,
    TILE_SIZE,
    TILE_STRIDE,
    TOP_SURFACE_FT,
)


def _grid() -> dict:
    m = np.load(GRID_META_PATH)
    return dict(
        x_min=float(m["x_min"]),
        y_max=float(m["y_max"]),
        resolution=float(m["resolution"]),
        rows=int(m["rows"]),
        cols=int(m["cols"]),
    )


def _cell_idx(x, y, g):
    c = np.clip(((x - g["x_min"]) / g["resolution"]).astype(np.int32), 0, g["cols"] - 1)
    r = np.clip(((g["y_max"] - y) / g["resolution"]).astype(np.int32), 0, g["rows"] - 1)
    return r, c


def pass1_surface_z(g, las_path=LAS_PATH, limit_chunks=None) -> np.ndarray:
    surface = np.full((g["rows"], g["cols"]), np.nan, dtype=np.float32)
    with laspy.open(las_path) as f:
        for i, ch in enumerate(f.chunk_iterator(CHUNK_SIZE)):
            r, c = _cell_idx(np.asarray(ch.x), np.asarray(ch.y), g)
            np.fmax.at(surface, (r, c), np.asarray(ch.z, dtype=np.float32))
            print(f"  pass1 chunk {i + 1}", flush=True)
            if limit_chunks and i + 1 >= limit_chunks:
                break
    return surface


def pass2_near_surface_means(g, surface, las_path=LAS_PATH, limit_chunks=None):
    """Accumulate per-cell means over points within TOP_SURFACE_FT of max Z."""
    shape = (g["rows"], g["cols"])
    sums = {
        k: np.zeros(shape, dtype=np.float64)
        for k in ("r", "g", "b", "exg", "hag", "inten")
    }
    count = np.zeros(shape, dtype=np.float64)

    exg_pts = np.load(DERIVED_DIR / "exg.npy", mmap_mode="r")
    hag_pts = np.load(DERIVED_DIR / "hag.npy", mmap_mode="r")

    off = 0
    with laspy.open(las_path) as f:
        for i, ch in enumerate(f.chunk_iterator(CHUNK_SIZE)):
            n = len(ch)
            r, c = _cell_idx(np.asarray(ch.x), np.asarray(ch.y), g)
            near = np.asarray(ch.z, dtype=np.float32) >= surface[r, c] - TOP_SURFACE_FT
            rn, cn = r[near], c[near]
            for key, vals in (
                ("r", np.asarray(ch.red, np.float64)[near]),
                ("g", np.asarray(ch.green, np.float64)[near]),
                ("b", np.asarray(ch.blue, np.float64)[near]),
                ("exg", np.asarray(exg_pts[off : off + n], np.float64)[near]),
                ("hag", np.asarray(hag_pts[off : off + n], np.float64)[near]),
                ("inten", np.asarray(ch.intensity, np.float64)[near]),
            ):
                np.add.at(sums[key], (rn, cn), vals)
            np.add.at(count, (rn, cn), 1.0)
            off += n
            print(f"  pass2 chunk {i + 1}", flush=True)
            if limit_chunks and i + 1 >= limit_chunks:
                break

    means = {}
    with np.errstate(invalid="ignore"):
        for key, s in sums.items():
            means[key] = np.divide(
                s, count, out=np.full(shape, np.nan), where=count > 0
            )
    return means, count


def write_tiles(rgb: np.ndarray, out_dir) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = rgb.shape[:2]

    def offsets(total):
        offs = list(range(0, max(total - TILE_SIZE, 0) + 1, TILE_STRIDE))
        last = max(total - TILE_SIZE, 0)
        if last not in offs:
            offs.append(last)
        return offs

    paths = []
    for r0 in offsets(H):
        for c0 in offsets(W):
            tile = rgb[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE]
            p = out_dir / f"tile_r{r0}_c{c0}.png"
            cv2.imwrite(str(p), tile[..., ::-1])  # RGB → BGR for cv2
            paths.append(str(p))
    return paths


def run_ortho(limit_chunks=None) -> None:
    g = _grid()
    SLICES_DIR.mkdir(parents=True, exist_ok=True)

    print("pass 1/2 — max-Z per cell …")
    surface = pass1_surface_z(g, limit_chunks=limit_chunks)
    void = np.isnan(surface)
    print(f"  {void.mean():.1%} void cells")

    print("pass 2/2 — near-surface means …")
    means, count = pass2_near_surface_means(g, surface, limit_chunks=limit_chunks)

    # 16-bit means → 8-bit RGB, voids inpainted (Telea)
    rgb = np.stack([means["r"], means["g"], means["b"]], axis=2)
    rgb = np.clip(np.nan_to_num(rgb) / RGB_16BIT_TO_8BIT_DIVISOR, 0, 255).astype(
        np.uint8
    )
    rgb = cv2.inpaint(rgb, void.astype(np.uint8), INPAINT_RADIUS_PX, cv2.INPAINT_TELEA)
    if (rgb[void].sum(axis=-1) == 0).any():
        raise RuntimeError(
            "black voids survived inpaint — cv2.inpaint failed to fill every "
            "void cell; check the void mask and inpaint radius"
        )

    cv2.imwrite(str(SLICES_DIR / "ortho_rgb.png"), rgb[..., ::-1])
    np.save(SLICES_DIR / "surface_z.npy", surface)
    np.save(SLICES_DIR / "exg_grid.npy", means["exg"].astype(np.float32))
    np.save(SLICES_DIR / "hag_grid.npy", means["hag"].astype(np.float32))
    np.save(SLICES_DIR / "intensity_grid.npy", means["inten"].astype(np.float32))
    np.save(SLICES_DIR / "void_mask.npy", void)

    tiles = write_tiles(rgb, SLICES_DIR / "tiles")
    print(f"Saved ortho_rgb.png + 5 grids + {len(tiles)} tiles → {SLICES_DIR}")
    print("INSPECT ortho_rgb.png: must look like an aerial photo (human gate).")


def _self_check() -> None:
    """Round-trip 1000 random world points through the grid mapping; assert
    tile offsets cover every pixel."""
    g = _grid()
    rng = np.random.default_rng(1)
    x = g["x_min"] + rng.uniform(0, g["cols"] * g["resolution"], 1000)
    y = g["y_max"] - rng.uniform(0, g["rows"] * g["resolution"], 1000)
    r, c = _cell_idx(x, y, g)
    x_back = g["x_min"] + (c + 0.5) * g["resolution"]
    y_back = g["y_max"] - (r + 0.5) * g["resolution"]
    assert np.all(np.abs(x - x_back) <= g["resolution"]), "x round-trip off"
    assert np.all(np.abs(y - y_back) <= g["resolution"]), "y round-trip off"

    for total in (g["rows"], g["cols"], TILE_SIZE, 2000):
        covered = np.zeros(total, dtype=bool)
        offs = list(range(0, max(total - TILE_SIZE, 0) + 1, TILE_STRIDE))
        last = max(total - TILE_SIZE, 0)
        if last not in offs:
            offs.append(last)
        for o in offs:
            covered[o : o + TILE_SIZE] = True
        assert covered.all(), f"tile gap for extent {total}"
    print("self-check OK: grid round-trip + tile coverage")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit-chunks", type=int, default=None)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        _self_check()
    else:
        run_ortho(limit_chunks=args.limit_chunks)
