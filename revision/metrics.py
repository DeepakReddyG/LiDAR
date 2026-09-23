"""Strict, array-only evaluation of the preregistered revision objective.

This module never opens data, creates reference labels, or infers geometry. The
caller supplies immutable reference arrays and any independently computed strata.
All fractions use [0, 1], not percentages. Undefined scores serialize as ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

REFERENCE_CODES = (3, 5, 6, 11, 64)
CLASS_NAMES = {3: "grass", 5: "tree", 6: "building", 11: "hard_surface", 64: "vehicle"}
IGNORED_REFERENCE_CODES = (0, 1)
UNLABELLED_PREDICTION_CODE = 1
PREDICTION_OUTCOMES = (*REFERENCE_CODES, "unlabelled", "invalid")
ROUTES = ("transfer", "fallback", "unknown")


def _vector(values, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} length {len(result)} does not match {length}")
    return result


def _ids(values, name: str) -> np.ndarray:
    result = _vector(values, name)
    if result.dtype.kind not in "iu" or (result < 0).any():
        raise ValueError(f"{name} must contain nonnegative integer IDs, not floats")
    result = result.astype(np.uint64, copy=False)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contains duplicate IDs")
    return result


def _divide(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _score(reference: np.ndarray, outcomes: np.ndarray) -> dict:
    """Score already aligned reviewed records, including empty optional strata."""
    confusion = np.zeros(
        (len(REFERENCE_CODES), len(PREDICTION_OUTCOMES)), dtype=np.int64
    )
    row = np.searchsorted(np.asarray(REFERENCE_CODES), reference)
    np.add.at(confusion, (row, outcomes), 1)
    support = confusion.sum(axis=1)
    n = int(support.sum())
    per_class = {}
    present_losses, present_ious = [], []
    for i, code in enumerate(REFERENCE_CODES):
        count = int(support[i])
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum()) - tp
        fn = count - tp
        present = count > 0
        loss = _divide(fn, count)
        iou = _divide(tp, tp + fp + fn) if present else None
        if present:
            present_losses.append(loss)
            present_ious.append(iou)
        per_class[str(code)] = {
            "name": CLASS_NAMES[code],
            "present": present,
            "support": count,
            "predicted_count": tp + fp,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "error_loss": loss,
            "precision": _divide(tp, tp + fp) if present else None,
            "recall": _divide(tp, count),
            "f1": _divide(2 * tp, 2 * tp + fp + fn) if present else None,
            "iou": iou,
            "unlabelled_count": int(confusion[i, -2]),
            "invalid_count": int(confusion[i, -1]),
        }
    correct = int(sum(confusion[i, i] for i in range(len(REFERENCE_CODES))))
    unlabelled = int(confusion[:, -2].sum())
    invalid = int(confusion[:, -1].sum())
    primary_loss = float(np.mean(present_losses)) if present_losses else None
    return {
        "scored_points": n,
        "correct_points": correct,
        "error_points": n - correct,
        "wrong_label_points": n - correct - unlabelled - invalid,
        "unlabelled_points": unlabelled,
        "invalid_points": invalid,
        "unlabelled_rate": _divide(unlabelled, n),
        "invalid_rate": _divide(invalid, n),
        "label_coverage": _divide(n - unlabelled - invalid, n),
        "primary_loss": primary_loss,
        "balanced_accuracy": 1.0 - primary_loss if primary_loss is not None else None,
        "overall_accuracy": _divide(correct, n),
        "macro_iou": float(np.mean(present_ious)) if present_ious else None,
        "present_reference_codes": [
            code for code, count in zip(REFERENCE_CODES, support) if count
        ],
        "confusion_matrix": confusion.tolist(),
        "reference_codes": list(REFERENCE_CODES),
        "prediction_outcomes": list(PREDICTION_OUTCOMES),
        "per_class": per_class,
    }


def evaluate_predictions(
    ref_ids,
    ref_codes,
    pred_ids,
    pred_codes,
    *,
    strata: Mapping[str, np.ndarray] | None = None,
    routes=None,
) -> dict:
    """Evaluate all expected reviewed IDs against predictions without dropping any.

    ``strata`` maps names (e.g. above_surface/below_surface/boundary) to boolean
    masks in *reference-array order*. Strata may overlap; each is reported
    separately. ``routes`` is an optional array in *prediction-array order*
    containing only transfer/fallback/unknown. Shares are reported both over all
    supplied predictions and over reviewed records; only the latter are scored.

    Codes 0/1 in reference arrays are unreviewed. Only prediction code 1 means
    unlabelled; every unsupported prediction code, including 0, is an invalid
    outcome costing one. Unknown reference codes fail rather than being silently
    excluded. Extra predictions and missing *unreviewed* records are recorded but
    never scored. Duplicate IDs in either complete input always fail.

    The caller remains responsible for exact coordinate/provenance verification.
    Identity matching alone does not prove that a point has unchanged geometry.
    """
    reference_ids = _ids(ref_ids, "ref_ids")
    prediction_ids = _ids(pred_ids, "pred_ids")
    reference = _vector(ref_codes, "ref_codes", len(reference_ids))
    prediction = _vector(pred_codes, "pred_codes", len(prediction_ids))
    if (
        reference.dtype.kind not in "iu"
        or not np.isin(reference, (*REFERENCE_CODES, *IGNORED_REFERENCE_CODES)).all()
    ):
        raise ValueError("ref_codes contains an unsupported reference code")
    if prediction.dtype.kind not in "iuf":
        raise ValueError(
            "pred_codes must be numeric; unsupported numeric values become invalid outcomes"
        )
    reviewed = np.isin(reference, REFERENCE_CODES)
    if not reviewed.any():
        raise ValueError("No reviewed reference points; no accuracy may be reported")

    order = np.argsort(prediction_ids)
    sorted_ids = prediction_ids[order]
    positions = np.searchsorted(sorted_ids, reference_ids)
    matched = positions < len(sorted_ids)
    # Only index in-range search positions: an empty/missing prediction set must
    # produce a completeness error, never an accidental indexing exception.
    matched[matched] = sorted_ids[positions[matched]] == reference_ids[matched]
    missing = reviewed & ~matched
    if missing.any():
        examples = reference_ids[missing][:5].tolist()
        raise ValueError(
            f"Missing prediction records for {int(missing.sum())} expected reviewed IDs; examples={examples}"
        )
    alignment = order[positions[reviewed]]
    scored_prediction = prediction[alignment]
    outcomes = np.full(
        len(scored_prediction), len(PREDICTION_OUTCOMES) - 1, dtype=np.int8
    )
    for i, code in enumerate(REFERENCE_CODES):
        outcomes[scored_prediction == code] = i
    outcomes[scored_prediction == UNLABELLED_PREDICTION_CODE] = (
        len(PREDICTION_OUTCOMES) - 2
    )
    scored_reference = reference[reviewed]
    result = _score(scored_reference, outcomes)
    result["schema_version"] = 1
    result["objective"] = {
        "name": "class_balanced_error_loss",
        "correct_cost": 0,
        "wrong_label_cost": 1,
        "unlabelled_cost": 1,
        "invalid_cost": 1,
        "macro_average": "reference classes present only",
    }
    result["integrity"] = {
        "reference_points": len(reference_ids),
        "prediction_points": len(prediction_ids),
        "expected_reviewed_points": int(reviewed.sum()),
        "matched_reviewed_points": int(reviewed.sum()),
        "missing_reviewed_points": 0,
        "unreviewed_reference_points": int((~reviewed).sum()),
        "missing_unreviewed_points": int((~reviewed & ~matched).sum()),
        "extra_prediction_points": len(prediction_ids) - int(matched.sum()),
        "geometry_checked": False,
    }
    result["strata"] = {}
    for name, mask in (strata or {}).items():
        if not isinstance(name, str):
            raise TypeError("stratum names must be strings")
        selection = _vector(mask, f"strata[{name}]", len(reference_ids))
        if selection.dtype.kind != "b":
            raise ValueError(f"strata[{name}] must be a boolean mask")
        selection = selection[reviewed]
        result["strata"][name] = _score(
            scored_reference[selection], outcomes[selection]
        )
        result["strata"][name]["share_of_reviewed_points"] = float(selection.mean())
    if routes is not None:
        route = _vector(routes, "routes", len(prediction_ids))
        if not np.isin(route, ROUTES).all():
            raise ValueError("routes must contain transfer, fallback or unknown")
        reviewed_route = route[alignment]
        result["routes"] = {}
        for name in ROUTES:
            selection = reviewed_route == name
            route_score = _score(scored_reference[selection], outcomes[selection])
            route_score["share_of_reviewed_points"] = float(selection.mean())
            route_score["all_prediction_points"] = int((route == name).sum())
            route_score["share_of_all_predictions"] = float((route == name).mean())
            result["routes"][name] = route_score
    return result


def aggregate_regions(region_results: Mapping[str, dict]) -> dict:
    """Equal-region macro loss per objective.md; never weight regions by density."""
    if not region_results:
        raise ValueError("At least one scored region is required")
    losses = [result["primary_loss"] for result in region_results.values()]
    if any(value is None or not np.isfinite(value) for value in losses):
        raise ValueError("Each region needs reviewed labels and a finite primary loss")
    loss = float(np.mean(losses))
    return {
        "primary_loss": loss,
        "balanced_accuracy": 1.0 - loss,
        "region_count": len(region_results),
        "aggregation": "equal region weights, each internally equal present-class weights",
        "region_primary_loss": {
            name: result["primary_loss"] for name, result in region_results.items()
        },
    }
