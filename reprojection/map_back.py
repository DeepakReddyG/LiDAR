"""reprojection/map_back.py — Z-aware label projection to 3D (MANUAL §6.6, T6).

The v1 column bug is dead: a 2D label applies only to points near the
visible surface of its pixel. Points below the surface (under canopy,
under eaves) are classified by LiDAR rules, not SAM3:

    on_surface = |z − surface_z[r, c]| < surface_ft
    below:  hag > below_tree_hag            → tree (trunk / understory)
            else exg > below_grass_exg      → grass
            else                            → pavement

Chunked end to end. Writes data/output/labelled.las with:
    classification  LAS codes per config.CLASSES (−1 → unassigned)
    user_data       confidence 0–255 (winning pixel confidence for surface
                    points, RULE_CONF for rule-labelled below-surface points)
and data/output/labels.npy — (N,) int32 memmap, LAS point order. (The v1
labelled_points.npz duplicated XYZ already stored in the LAS; labels.npy +
the LAS carry everything.)
"""

from __future__ import annotations

import laspy
import numpy as np

from config import (CHUNK_SIZE, CLASSES, DERIVED_DIR, GRID_META_PATH,
                    LAS_PATH, MAP_BACK, MASKS_DIR, OUTPUT_DIR, SLICES_DIR,
                    UNLABELLED_LAS_CODE)

_NAME_TO_ID = {info["name"]: cid for cid, info in CLASSES.items()}


def label_points(z, r, c, label_grid, conf_grid, surface_z, hag, exg):
    """Pure core: label one batch of points. Returns (labels, conf_u8)."""
    m = MAP_BACK
    surf = surface_z[r, c]
    with np.errstate(invalid="ignore"):
        on_surface = np.abs(z - surf) < m["surface_ft"]   # NaN surf → False

    labels = np.where(on_surface, label_grid[r, c], -1).astype(np.int32)
    conf = np.where(on_surface, (conf_grid[r, c] * 255), 0).astype(np.uint8)

    below = ~on_surface
    tree = below & (hag > m["below_tree_hag"])
    grass = below & ~tree & (exg > m["below_grass_exg"])
    pave = below & ~tree & ~grass
    labels[tree] = _NAME_TO_ID["tree"]
    labels[grass] = _NAME_TO_ID["grass"]
    labels[pave] = _NAME_TO_ID["pavement"]
    conf[below] = m["rule_conf"]
    return labels, conf


def run_map_back() -> None:
    meta = np.load(GRID_META_PATH)
    x_min, y_max = float(meta["x_min"]), float(meta["y_max"])
    res, rows, cols = float(meta["resolution"]), int(meta["rows"]), int(meta["cols"])

    label_grid = np.load(MASKS_DIR / "label_grid.npy")
    conf_grid = np.load(MASKS_DIR / "conf_grid.npy")
    surface_z = np.load(SLICES_DIR / "surface_z.npy")
    hag_pts = np.load(DERIVED_DIR / "hag.npy", mmap_mode="r")
    exg_pts = np.load(DERIVED_DIR / "exg.npy", mmap_mode="r")

    id_to_code = {cid: info["las_code"] for cid, info in CLASSES.items()}
    id_to_code[-1] = UNLABELLED_LAS_CODE
    code_lut = np.full(max(_NAME_TO_ID.values()) + 2, UNLABELLED_LAS_CODE, np.uint8)
    for cid, code in id_to_code.items():
        code_lut[cid] = code                    # index −1 wraps to the last slot

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "labelled.las"

    with laspy.open(LAS_PATH) as reader:
        n_pts = reader.header.point_count
        all_labels = np.lib.format.open_memmap(
            OUTPUT_DIR / "labels.npy", mode="w+", dtype=np.int32, shape=(n_pts,))
        counts = np.zeros(len(code_lut), dtype=np.int64)

        with laspy.open(out_path, mode="w", header=reader.header) as writer:
            off = 0
            for i, ch in enumerate(reader.chunk_iterator(CHUNK_SIZE)):
                n = len(ch)
                x, y, z = np.asarray(ch.x), np.asarray(ch.y), np.asarray(ch.z, np.float32)
                c = np.clip(((x - x_min) / res).astype(np.int32), 0, cols - 1)
                r = np.clip(((y_max - y) / res).astype(np.int32), 0, rows - 1)

                labels, conf = label_points(
                    z, r, c, label_grid, conf_grid, surface_z,
                    np.asarray(hag_pts[off:off + n]),
                    np.asarray(exg_pts[off:off + n]))

                ch.classification = code_lut[labels]
                ch.user_data = conf
                writer.write_points(ch)

                all_labels[off:off + n] = labels
                np.add.at(counts, labels, 1)
                off += n
                print(f"  chunk {i + 1}: {off:,}/{n_pts:,}", flush=True)
        all_labels.flush()

    print("\nPoint-cloud class distribution:")
    for name, cid in _NAME_TO_ID.items():
        print(f"  {name:<10s} {counts[cid]:>13,}  ({counts[cid] / n_pts:.2%})")
    print(f"  unlabelled {counts[-1]:>13,}  ({counts[-1] / n_pts:.2%})")
    print(f"Saved {out_path} (+ labels.npy)")


def _self_check() -> None:
    """One labelled tree pixel; a vertical column of points over it."""
    label_grid = np.full((4, 4), -1, np.int32)
    label_grid[1, 1] = _NAME_TO_ID["tree"]
    conf_grid = np.zeros((4, 4), np.float32); conf_grid[1, 1] = 0.8
    surface_z = np.full((4, 4), np.nan, np.float32); surface_z[1, 1] = 30.0

    #        canopy   trunk   grass-ground  bare-ground
    z = np.array([29.0,   15.0,   0.5,          0.5], np.float32)
    hag = np.array([29.0,  15.0,   0.5,          0.5], np.float32)
    exg = np.array([0.3,   0.1,    0.2,          0.0], np.float32)
    r = np.array([1, 1, 1, 1]); c = np.array([1, 1, 1, 1])

    labels, conf = label_points(z, r, c, label_grid, conf_grid, surface_z, hag, exg)
    want = [_NAME_TO_ID["tree"], _NAME_TO_ID["tree"],
            _NAME_TO_ID["grass"], _NAME_TO_ID["pavement"]]
    assert labels.tolist() == want, f"{labels.tolist()} != {want}"
    assert conf[0] == np.uint8(0.8 * 255) and conf[1] == 128
    # NaN surface (void pixel) → below-surface rules, never the 2D label
    labels2, _ = label_points(z[:1], np.array([0]), np.array([0]),
                              label_grid, conf_grid, surface_z, hag[:1], exg[:1])
    assert labels2[0] == _NAME_TO_ID["tree"]     # hag 29 → tree by rule
    print("self-check OK: canopy=tree, trunk=tree, ground split by ExG, void→rules")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--self-check", action="store_true")
    if p.parse_args().self_check:
        _self_check()
    else:
        run_map_back()
