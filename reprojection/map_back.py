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
    user_data       unchanged source attribute
    model_score     float32 winning class's pixel score, not calibrated confidence;
                    NaN for rule fallback or unlabelled points
    prediction_source uint8 route enum (see PREDICTION_SOURCES)
    internal_class_id int16 class ID before the lossy LAS class-code mapping
and data/output/labels.npy — (N,) int32 memmap, LAS point order. (The v1
labelled_points.npz duplicated XYZ already stored in the LAS; labels.npy +
the LAS carry everything.)
"""

from __future__ import annotations

import laspy
import numpy as np

from config import (
    CHUNK_SIZE,
    CLASSES,
    DERIVED_DIR,
    GRID_META_PATH,
    LAS_PATH,
    MAP_BACK,
    MASKS_DIR,
    OUTPUT_DIR,
    SLICES_DIR,
    UNLABELLED_LAS_CODE,
)

_NAME_TO_ID = {info["name"]: cid for cid, info in CLASSES.items()}
PREDICTION_SOURCES = {
    0: "surface_grid_unlabelled",
    1: "surface_grid_transfer",
    2: "below_surface_rule",
    3: "missing_surface_rule",
    4: "above_surface_rule",
}
PREDICTION_DIMENSIONS = {
    "model_score": (np.dtype("float32"), "Mask score; NaN for no model"),
    "prediction_source": (np.dtype("uint8"), "0 none;1 grid;2 low;3 void;4 up"),
    "internal_class_id": (np.dtype("int16"), "Semantic ID; -1 unlabelled"),
}


def label_points(z, r, c, label_grid, conf_grid, surface_z, hag, exg):
    """Legacy replay API: return (labels, mixed score/flag uint8).

    The second result is preserved for historical reproduction only. New
    exports use label_points_with_provenance and retain source user_data.
    """
    m = MAP_BACK
    surf = surface_z[r, c]
    with np.errstate(invalid="ignore"):
        on_surface = np.abs(z - surf) < m["surface_ft"]  # NaN surf → False

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


def label_points_with_provenance(z, r, c, label_grid, conf_grid, surface_z, hag, exg):
    """Return legacy-identical labels, model scores, and explicit route codes.

    A surface-grid label may have zero score if smoothing propagated a class
    unsupported by a surviving mask at that pixel. Such zeroes are retained;
    neither they nor nonzero mask scores are calibrated probabilities.
    """
    labels, _legacy_flags = label_points(
        z, r, c, label_grid, conf_grid, surface_z, hag, exg
    )
    surf = surface_z[r, c]
    with np.errstate(invalid="ignore"):
        on_surface = np.abs(z - surf) < MAP_BACK["surface_ft"]
    sources = np.full(labels.shape, 2, dtype=np.uint8)
    sources[~np.isfinite(surf)] = 3
    sources[np.isfinite(surf) & ~on_surface & (z > surf)] = 4
    sources[on_surface & (labels < 0)] = 0
    supported = on_surface & (labels >= 0)
    sources[supported] = 1
    scores = np.full(labels.shape, np.nan, dtype=np.float32)
    scores[supported] = conf_grid[r[supported], c[supported]]
    return labels, scores, sources


def prediction_header(source_header):
    """Copy original header/CRS/VLR metadata and declare separate output fields.

    A collision is refused even for an apparently compatible dtype: silently
    replacing an existing source attribute would lose its original meaning.
    """
    collisions = set(source_header.point_format.dimension_names) & set(
        PREDICTION_DIMENSIONS
    )
    if collisions:
        raise ValueError(f"Prediction dimension collision: {sorted(collisions)}")
    header = source_header.copy()
    header.add_extra_dims(
        [
            laspy.ExtraBytesParams(name=name, type=dtype, description=description)
            for name, (dtype, description) in PREDICTION_DIMENSIONS.items()
        ]
    )
    return header


def prediction_records(points, header, labels, model_scores, sources):
    """Copy one source chunk exactly, replacing only classification/new fields.

    Input and output remain separate. Copying packed structured fields preserves
    source flags, integer XYZ and scaled extra dimensions without conversion.
    """
    n = len(points)
    labels, model_scores, sources = map(np.asarray, (labels, model_scores, sources))
    if any(values.shape != (n,) for values in (labels, model_scores, sources)):
        raise ValueError("Prediction fields must have one value per source point")
    if not np.isin(labels, [-1, *_NAME_TO_ID.values()]).all():
        raise ValueError("Unknown internal class ID")
    if not np.isin(sources, list(PREDICTION_SOURCES)).all():
        raise ValueError("Unknown prediction source")
    expected = set(PREDICTION_DIMENSIONS)
    if not expected.issubset(header.point_format.dimension_names):
        raise ValueError("Output header must be created with prediction_header")
    if expected.intersection(points.point_format.dimension_names):
        raise ValueError("Prediction dimension collision in source points")
    if hasattr(points, "scales") and (
        not np.array_equal(points.scales, header.scales)
        or not np.array_equal(points.offsets, header.offsets)
    ):
        raise ValueError("Output scales/offsets must match source records")
    output = laspy.ScaleAwarePointRecord.zeros(n, header=header)
    for name in points.array.dtype.names:
        if name not in output.array.dtype.names:
            raise ValueError(f"Source field absent from output header: {name}")
        output.array[name] = points.array[name]
    output.classification = labels_to_las_codes(labels)
    output.model_score = model_scores.astype(np.float32, copy=False)
    output.prediction_source = sources.astype(np.uint8, copy=False)
    output.internal_class_id = labels.astype(np.int16, copy=False)
    return output


def labels_to_las_codes(labels: np.ndarray) -> np.ndarray:
    """Map class ids to LAS codes; −1 → UNLABELLED_LAS_CODE (no index wrap)."""
    max_id = max(_NAME_TO_ID.values())
    code_lut = np.full(max_id + 1, UNLABELLED_LAS_CODE, np.uint8)
    for cid, info in CLASSES.items():
        code_lut[cid] = info["las_code"]
    out = np.full(labels.shape, UNLABELLED_LAS_CODE, np.uint8)
    valid = labels >= 0
    out[valid] = code_lut[labels[valid]]
    return out


def run_map_back() -> None:
    meta = np.load(GRID_META_PATH)
    x_min, y_max = float(meta["x_min"]), float(meta["y_max"])
    res, rows, cols = float(meta["resolution"]), int(meta["rows"]), int(meta["cols"])

    label_grid = np.load(MASKS_DIR / "label_grid.npy")
    conf_grid = np.load(MASKS_DIR / "conf_grid.npy")
    surface_z = np.load(SLICES_DIR / "surface_z.npy")
    hag_pts = np.load(DERIVED_DIR / "hag.npy", mmap_mode="r")
    exg_pts = np.load(DERIVED_DIR / "exg.npy", mmap_mode="r")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "labelled.las"
    if out_path.exists() or (OUTPUT_DIR / "labels.npy").exists():
        raise FileExistsError(
            "Preserve previous predictions; use a new output directory or archive them explicitly"
        )

    with laspy.open(LAS_PATH) as reader:
        n_pts = reader.header.point_count
        out_header = prediction_header(reader.header)
        all_labels = np.lib.format.open_memmap(
            OUTPUT_DIR / "labels.npy", mode="w+", dtype=np.int32, shape=(n_pts,)
        )
        counts = {cid: 0 for cid in _NAME_TO_ID.values()}
        n_unlabelled = 0

        with laspy.open(out_path, mode="w", header=out_header) as writer:
            off = 0
            for i, ch in enumerate(reader.chunk_iterator(min(CHUNK_SIZE, 500_000))):
                n = len(ch)
                x, y, z = (
                    np.asarray(ch.x),
                    np.asarray(ch.y),
                    np.asarray(ch.z, np.float32),
                )
                c = np.clip(((x - x_min) / res).astype(np.int32), 0, cols - 1)
                r = np.clip(((y_max - y) / res).astype(np.int32), 0, rows - 1)

                labels, scores, sources = label_points_with_provenance(
                    z,
                    r,
                    c,
                    label_grid,
                    conf_grid,
                    surface_z,
                    np.asarray(hag_pts[off : off + n]),
                    np.asarray(exg_pts[off : off + n]),
                )

                writer.write_points(
                    prediction_records(ch, out_header, labels, scores, sources)
                )

                all_labels[off : off + n] = labels
                for cid in counts:
                    counts[cid] += int((labels == cid).sum())
                n_unlabelled += int((labels < 0).sum())
                off += n
                print(f"  chunk {i + 1}: {off:,}/{n_pts:,}", flush=True)
        all_labels.flush()

    print("\nPoint-cloud class distribution:")
    for name, cid in _NAME_TO_ID.items():
        print(f"  {name:<10s} {counts[cid]:>13,}  ({counts[cid] / n_pts:.2%})")
    print(f"  unlabelled {n_unlabelled:>13,}  ({n_unlabelled / n_pts:.2%})")
    print(f"Saved {out_path} (+ labels.npy)")


def _self_check() -> None:
    """One labelled tree pixel; a vertical column of points over it."""
    label_grid = np.full((4, 4), -1, np.int32)
    label_grid[1, 1] = _NAME_TO_ID["tree"]
    conf_grid = np.zeros((4, 4), np.float32)
    conf_grid[1, 1] = 0.8
    surface_z = np.full((4, 4), np.nan, np.float32)
    surface_z[1, 1] = 30.0

    #        canopy   trunk   grass-ground  bare-ground
    z = np.array([29.0, 15.0, 0.5, 0.5], np.float32)
    hag = np.array([29.0, 15.0, 0.5, 0.5], np.float32)
    exg = np.array([0.3, 0.1, 0.2, 0.0], np.float32)
    r = np.array([1, 1, 1, 1])
    c = np.array([1, 1, 1, 1])

    labels, conf = label_points(z, r, c, label_grid, conf_grid, surface_z, hag, exg)
    want = [
        _NAME_TO_ID["tree"],
        _NAME_TO_ID["tree"],
        _NAME_TO_ID["grass"],
        _NAME_TO_ID["pavement"],
    ]
    assert labels.tolist() == want, f"{labels.tolist()} != {want}"
    assert conf[0] == np.uint8(0.8 * 255) and conf[1] == 128
    # NaN surface (void pixel) → below-surface rules, never the 2D label
    labels2, _ = label_points(
        z[:1],
        np.array([0]),
        np.array([0]),
        label_grid,
        conf_grid,
        surface_z,
        hag[:1],
        exg[:1],
    )
    assert labels2[0] == _NAME_TO_ID["tree"]  # hag 29 → tree by rule

    codes = labels_to_las_codes(np.array([-1, _NAME_TO_ID["tree"]], np.int32))
    assert codes[0] == UNLABELLED_LAS_CODE
    assert codes[1] == CLASSES[_NAME_TO_ID["tree"]]["las_code"]
    print(
        "self-check OK: canopy=tree, trunk=tree, ground split by ExG, void→rules, LUT"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self-check", action="store_true")
    if p.parse_args().self_check:
        _self_check()
    else:
        run_map_back()
