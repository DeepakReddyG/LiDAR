"""Pure array components for registered, matched development comparisons.

No configuration module, data file, cache or reference annotation is accessed.
The inherited defaults deliberately preserve its labels, including majority
smoothing's possible propagation into initially excluded pixels. Alternative
switches isolate named operations; they do not silently repair other stages.
"""

from __future__ import annotations

import numpy as np


def _classes(parameters):
    classes = {int(key): dict(value) for key, value in parameters["CLASSES"].items()}
    names = {value["name"]: key for key, value in classes.items()}
    if len(names) != len(classes) or not classes or min(classes) < 0:
        raise ValueError("Classes require unique names and nonnegative internal IDs")
    return classes, names


def _same_shape(*arrays):
    arrays = tuple(np.asarray(value) for value in arrays)
    if not arrays or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("Array shapes must match")
    return arrays


def majority_filter(grid, size):
    """Inherited edge-padded mode: ties select the smallest internal class ID.

    Unlabelled -1 participates in voting. Even sizes use the inherited asymmetric
    padding convention; the registered experiment uses odd sizes only.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or not grid.size:
        raise ValueError("Majority filter requires a nonempty 2D label grid")
    if (
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 1
    ):
        raise ValueError("Majority window must be a positive integer")
    height, width = grid.shape
    lo, hi = size // 2, size - size // 2 - 1
    best_count = np.full(grid.shape, -1, np.int32)
    best = grid.copy()
    for value in np.unique(grid):
        padded = np.pad(
            (grid == value).astype(np.int32), ((lo, hi), (lo, hi)), mode="edge"
        )
        sums = np.zeros((height + size, width + size), np.int32)
        sums[1:, 1:] = padded.cumsum(0).cumsum(1)
        counts = (
            sums[size : size + height, size : size + width]
            - sums[:height, size : size + width]
            - sums[size : size + height, :width]
            + sums[:height, :width]
        )
        take = counts > best_count
        best_count = np.where(take, counts, best_count)
        best = np.where(take, value, best)
    return best.astype(grid.dtype)


def _veto(name, exg, hag, void, parameters):
    veto = parameters["VETO"]
    with np.errstate(invalid="ignore"):
        if name == "tree":
            keep = (exg > veto["tree_exg"]) | (hag > veto["tree_hag"])
        elif name == "grass":
            keep = (exg > veto["grass_exg"]) & (hag < veto["grass_hag"])
        elif name == "building":
            keep = hag > veto["building_hag"]
        elif name in ("pavement", "sidewalk", "parking"):
            keep = hag < veto["pavement_hag"]
        elif name == "vehicle":
            keep = (hag > veto["vehicle_hag"][0]) & (hag < veto["vehicle_hag"][1])
        else:
            raise ValueError(f"No registered height/color rule for {name}")
    return keep & ~void


def fuse_components(
    conf,
    exg_grid,
    hag_grid,
    void,
    parameters,
    *,
    constraints=True,
    smoothing=True,
    majority_size=None,
    tree_height_check=False,
    physical_smoothing=False,
):
    """Fuse proposals with explicit configuration and independently named stages.

    Disabling constraints removes only height/color rules: pre-smoothing void
    exclusion, thresholds, priority and vocabulary remain matched. The special
    physical_smoothing and tree_height_check switches reproduce the historical
    experiments even when other controls are disabled. Returned score is the
    supplied winning-class score at each pixel; it is zero on unlabelled pixels.
    Rule-proposal scores are binary support, never calibrated model confidence.
    """
    classes, names = _classes(parameters)
    exg, hag, void = _same_shape(exg_grid, hag_grid, void)
    if exg.ndim != 2 or not exg.size or void.dtype.kind != "b":
        raise ValueError("Fusion requires nonempty 2D grids and a boolean void mask")
    if (
        set(conf) != set(names)
        or set(parameters["FUSE_PRIORITY"]) != set(names)
        or len(parameters["FUSE_PRIORITY"]) != len(names)
    ):
        raise ValueError(
            "Proposals and priority must include each configured class exactly once"
        )
    proposals = {name: np.asarray(value) for name, value in conf.items()}
    if any(
        value.shape != exg.shape
        or not np.isfinite(value).all()
        or (value < 0).any()
        or (value > 1).any()
        for value in proposals.values()
    ):
        raise ValueError("Proposal scores must match the grid and be finite in [0,1]")
    thresholds = {value["name"]: value["threshold"] for value in classes.values()}
    label = np.full(exg.shape, -1, np.int32)
    stats = {}
    tree_allowed = np.isfinite(hag) & (hag > parameters["VETO"]["tree_hag"]) & ~void
    for name in parameters["FUSE_PRIORITY"]:
        claimed = proposals[name] >= thresholds[name]
        kept = claimed & (
            _veto(name, exg, hag, void, parameters) if constraints else ~void
        )
        if tree_height_check and name == "tree":
            kept &= tree_allowed
        n_claim, n_kept = int(claimed.sum()), int(kept.sum())
        stats[name] = {
            "claimed_px": n_claim,
            "kept_px": n_kept,
            "veto_rejection": round(1 - n_kept / n_claim, 3) if n_claim else None,
        }
        label[(label == -1) & kept] = names[name]
    before = label
    if smoothing:
        window = (
            parameters["MAJORITY_FILTER_SIZE"]
            if majority_size is None
            else majority_size
        )
        label = majority_filter(label, window)
    else:
        label = label.copy()
    if physical_smoothing:
        accept = np.zeros(exg.shape, bool)
        for name, class_id in names.items():
            allowed = _veto(name, exg, hag, void, parameters)
            if tree_height_check and name == "tree":
                allowed &= tree_allowed
            accept |= (label == class_id) & allowed
        label = np.where(accept, label, before)
    elif tree_height_check:
        invalid_tree = (label == names["tree"]) & ~tree_allowed
        label[invalid_tree] = before[invalid_tree]
    score = np.zeros(exg.shape, np.float32)
    for name, class_id in names.items():
        selected = label == class_id
        score[selected] = proposals[name][selected]
    return label, score, stats


def classify_rules(exg, hag, parameters):
    """The unchanged pointwise geometry+RGB baseline, returning internal IDs."""
    exg, hag = _same_shape(exg, hag)
    _, names = _classes(parameters)
    rules = parameters["BASELINE"]
    labels = np.full(exg.shape, -1, np.int32)
    choices = [
        ("building", (hag > rules["building_hag"]) & (exg <= rules["hard_exg"])),
        ("tree", (hag > rules["tree_hag"]) & (exg > rules["veg_exg"])),
        ("grass", (hag < rules["grass_hag"]) & (exg > rules["veg_exg"])),
        ("pavement", (hag < rules["pavement_hag"]) & (exg <= rules["hard_exg"])),
    ]
    for name, selected in choices:
        labels[(labels == -1) & selected] = names[name]
    return labels


def rules_confidence(exg, hag, parameters):
    """All configured class proposals, with one-hot standalone-rule support.

    Sidewalk, parking and vehicle remain zero because the standalone baseline
    does not distinguish them. This tests rule proposals, not model calibration.
    """
    _, names = _classes(parameters)
    labels = classify_rules(exg, hag, parameters)
    return {
        name: (labels == class_id).astype(np.float32)
        for name, class_id in names.items()
    }


def geometry_only(hag, parameters):
    """Restricted tree/grass proxy from the fixed plan, not a full-class model."""
    hag = np.asarray(hag)
    _, names = _classes(parameters)
    baseline = parameters["BASELINE"]
    labels = np.full(hag.shape, -1, np.int32)
    finite = np.isfinite(hag)
    labels[finite & (hag > baseline["tree_hag"])] = names["tree"]
    labels[finite & (hag < baseline["grass_hag"])] = names["grass"]
    return labels


def labels_to_las_codes(labels, parameters):
    """Explicit output crosswalk; unsupported internal IDs fail, not wrap."""
    labels = np.asarray(labels)
    classes, _ = _classes(parameters)
    if labels.dtype.kind not in "iu" or not np.isin(labels, [-1, *classes]).all():
        raise ValueError("Unsupported internal class ID")
    result = np.full(labels.shape, parameters["UNLABELLED_LAS_CODE"], np.uint8)
    for class_id, info in classes.items():
        result[labels == class_id] = info["las_code"]
    return result


def transfer_components(
    z,
    r,
    c,
    grid,
    score,
    surface,
    hag,
    exg,
    parameters,
    *,
    mode="height_aware",
    band=None,
):
    """Transfer labels and keep score separate from the assignment mechanism.

    Heights are cast to float32 to match the inherited pipeline's point-transfer
    call. ``model_score`` stores the supplied pixel score only for labelled
    transferred points; fallback and unlabelled outcomes are NaN. For rules as
    source, this field represents binary proposal support, not a neural score.

    Source codes: 0 unlabelled transfer attempt; 1 labelled pixel transfer;
    2 below-surface fallback; 3 missing-surface fallback; 4 above-surface fallback.
    Routes describe attempts, so unlabelled pixel transfers still count toward
    transfer share. Naive mode transfers the pixel at every height without rules.
    """
    if mode not in {"naive", "height_aware"}:
        raise ValueError("Transfer mode must be naive or height_aware")
    z, r, c, hag, exg = _same_shape(z, r, c, hag, exg)
    grid, score, surface = _same_shape(grid, score, surface)
    if z.ndim != 1 or grid.ndim != 2 or not grid.size:
        raise ValueError("Transfer requires point vectors and a nonempty 2D grid")
    if (
        r.dtype.kind not in "iu"
        or c.dtype.kind not in "iu"
        or (r < 0).any()
        or (c < 0).any()
        or (r >= grid.shape[0]).any()
        or (c >= grid.shape[1]).any()
    ):
        raise ValueError("Point row/column indices are outside the grid")
    classes, names = _classes(parameters)
    if grid.dtype.kind not in "iu" or not np.isin(grid, [-1, *classes]).all():
        raise ValueError("Unsupported internal class ID in pixel grid")
    config = parameters["MAP_BACK"]
    threshold = config["surface_ft"] if band is None else band
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Transfer band must be finite and positive")
    delta = np.asarray(z, dtype=np.float32) - surface[r, c]
    finite = np.isfinite(surface[r, c])
    transfer = (
        np.ones(len(z), bool)
        if mode == "naive"
        else finite & (np.abs(delta) < threshold)
    )
    labels = np.where(transfer, grid[r, c], -1).astype(np.int32)
    model_score = np.full(len(z), np.nan, np.float32)
    labelled_transfer = transfer & (labels >= 0)
    values = score[r[labelled_transfer], c[labelled_transfer]]
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Transferred proposal scores must be finite in [0,1]")
    model_score[labelled_transfer] = values
    prediction_source = np.zeros(len(z), np.uint8)
    prediction_source[labelled_transfer] = 1
    fallback = ~transfer
    tree = fallback & (hag > config["below_tree_hag"])
    grass = fallback & ~tree & (exg > config["below_grass_exg"])
    pavement = fallback & ~tree & ~grass
    labels[tree] = names["tree"]
    labels[grass] = names["grass"]
    labels[pavement] = names["pavement"]
    prediction_source[fallback & finite & (delta <= 0)] = 2
    prediction_source[fallback & ~finite] = 3
    prediction_source[fallback & finite & (delta > 0)] = 4
    if not np.isfinite(z).all():
        raise ValueError("Point heights must be finite")
    return {
        "labels": labels,
        "model_score": model_score,
        "prediction_source": prediction_source,
        "routes": np.where(transfer, "transfer", "fallback"),
    }
