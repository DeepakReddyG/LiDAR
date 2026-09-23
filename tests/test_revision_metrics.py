"""Synthetic scoring-contract checks; no survey or holdout data is read."""

import json

import numpy as np
import pytest

from revision.metrics import aggregate_regions, evaluate_predictions


def score(reference, prediction, **kwargs):
    ids = np.arange(len(reference), dtype=np.uint64)
    return evaluate_predictions(ids, reference, ids, prediction, **kwargs)


def test_expected_reviewed_id_missing_is_a_hard_failure():
    with pytest.raises(
        ValueError, match="Missing prediction records for 1 expected reviewed"
    ):
        evaluate_predictions([10, 11, 12], [3, 1, 5], [10, 11], [3, 1])
    with pytest.raises(ValueError, match="Missing prediction records"):
        evaluate_predictions([10], [3], np.array([], dtype=np.uint64), [])


@pytest.mark.parametrize("side", ["ref", "pred"])
def test_duplicate_ids_rejected_even_if_unreviewed(side):
    ref = [10, 10] if side == "ref" else [10, 11]
    pred = [10, 10] if side == "pred" else [10, 11]
    with pytest.raises(ValueError, match="duplicate IDs"):
        evaluate_predictions(ref, [3, 1], pred, [3, 1])


def test_shuffled_high_precision_integer_ids_align_without_mutation():
    ids = np.array([2**63 + 11, 2**63 + 12], dtype=np.uint64)
    reference = np.array([3, 5], dtype=np.uint8)
    prediction = np.array([5, 3], dtype=np.uint8)
    reference.flags.writeable = prediction.flags.writeable = False
    result = evaluate_predictions(ids, reference, ids[::-1], prediction)
    assert result["overall_accuracy"] == 1
    assert result["primary_loss"] == 0
    assert reference.tolist() == [3, 5]
    assert prediction.tolist() == [5, 3]


def test_float_ids_rejected_instead_of_risking_large_id_corruption():
    with pytest.raises(ValueError, match="integer IDs, not floats"):
        evaluate_predictions([1.0], [3], [1], [3])


def test_unreviewed_excluded_missing_unreviewed_allowed_extra_predictions_ignored():
    result = evaluate_predictions(
        [10, 11, 12, 13], [3, 0, 1, 5], [13, 10, 11, 99], [5, 3, 64, 5]
    )
    assert result["scored_points"] == 2
    assert result["overall_accuracy"] == 1
    assert result["integrity"]["unreviewed_reference_points"] == 2
    assert result["integrity"]["missing_unreviewed_points"] == 1
    assert result["integrity"]["extra_prediction_points"] == 1


def test_unknown_reference_never_silently_excluded_and_no_reference_cannot_score():
    with pytest.raises(ValueError, match="unsupported reference code"):
        score([3, 2], [3, 3])
    with pytest.raises(ValueError, match="No reviewed reference points"):
        score([0, 1], [3, 5])


def test_full_rectangular_matrix_invalid_and_unlabelled_equal_wrong_label_cost():
    result = score([3, 3, 3, 3, 5, 5], [3, 1, 0, 6, 5, np.nan])
    matrix = np.asarray(result["confusion_matrix"])
    assert matrix.shape == (5, 7)
    assert matrix[0].tolist() == [1, 0, 1, 0, 0, 1, 1]
    assert matrix[1].tolist() == [0, 1, 0, 0, 0, 0, 1]
    assert result["wrong_label_points"] == 1
    assert result["unlabelled_points"] == 1
    assert result["invalid_points"] == 2
    assert result["primary_loss"] == pytest.approx((3 / 4 + 1 / 2) / 2)
    assert result["overall_accuracy"] == pytest.approx(2 / 6)
    assert result["per_class"]["6"]["present"] is False
    assert result["per_class"]["6"]["false_positive"] == 1
    assert result["per_class"]["6"]["iou"] is None
    assert result["macro_iou"] == pytest.approx((1 / 4 + 1 / 2) / 2)
    json.dumps(result, allow_nan=False)


def test_preregistered_class_balancing_can_reverse_accuracy_ranking():
    # Majority class has 9 points. A predicts only that class: 90% overall but
    # 50% balanced accuracy. B recovers the minority while losing two majority.
    reference = [3] * 9 + [5]
    a = score(reference, [3] * 10)
    b = score(reference, [5, 5] + [3] * 7 + [5])
    assert a["overall_accuracy"] > b["overall_accuracy"]
    assert a["primary_loss"] > b["primary_loss"]


def test_abstention_cannot_improve_primary_loss_when_wrong_labels_are_removed():
    wrong = score([3, 5], [5, 5])
    abstain = score([3, 5], [1, 5])
    assert wrong["primary_loss"] == abstain["primary_loss"]
    assert wrong["macro_iou"] < abstain["macro_iou"]


def test_precision_recall_f1_iou_count_other_class_and_out_of_taxonomy_errors():
    result = score([3, 3, 5, 5], [3, 5, 3, 1])
    grass = result["per_class"]["3"]
    assert grass["precision"] == 0.5
    assert grass["recall"] == 0.5
    assert grass["f1"] == 0.5
    assert grass["iou"] == pytest.approx(1 / 3)
    assert result["per_class"]["5"]["iou"] == 0


def test_strata_and_routes_align_and_retain_empty_strata_without_fake_scores():
    result = evaluate_predictions(
        [10, 11, 12, 13],
        [3, 3, 1, 5],
        [13, 12, 11, 10],
        [5, 64, 1, 3],
        strata={
            "above_surface": np.array([False, False, False, True]),
            "below_surface": np.array([True, True, False, False]),
            "boundary": np.array([True, False, False, True]),
            "empty": np.zeros(4, dtype=bool),
        },
        routes=["transfer", "fallback", "fallback", "transfer"],
    )
    assert result["strata"]["above_surface"]["overall_accuracy"] == 1
    assert result["strata"]["below_surface"]["overall_accuracy"] == 0.5
    assert result["strata"]["boundary"]["scored_points"] == 2
    assert result["strata"]["empty"]["primary_loss"] is None
    assert result["routes"]["transfer"]["share_of_reviewed_points"] == pytest.approx(
        2 / 3
    )
    assert result["routes"]["fallback"]["share_of_all_predictions"] == 0.5
    assert result["routes"]["fallback"]["scored_points"] == 1
    assert result["routes"]["fallback"]["primary_loss"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"strata": {"bad": [1, 0]}}, "boolean mask"),
        ({"strata": {"bad": [True]}}, "length"),
        ({"routes": ["transfer", "invented"]}, "transfer, fallback or unknown"),
    ],
)
def test_ambiguous_stratum_or_route_inputs_fail(kwargs, match):
    with pytest.raises(ValueError, match=match):
        score([3, 5], [3, 5], **kwargs)


def test_region_aggregation_weights_regions_equally_not_point_density():
    many = score([3] * 100, [3] * 100)
    small = score([3], [5])
    result = aggregate_regions({"large": many, "small": small})
    assert result["primary_loss"] == 0.5
    with pytest.raises(ValueError, match="At least one"):
        aggregate_regions({})
