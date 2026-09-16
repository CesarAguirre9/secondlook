"""Unit tests for modeling.metrics: val-selected thresholds, per-dataset
metrics, and the dataset-prior shortcut baseline.

Pure numpy/sklearn -- no TensorFlow, no data -- so they run anywhere in ms.

The safety-critical assertions here are:
  * a threshold chosen on one split, applied to another, reports the sensitivity
    it ACTUALLY gets there -- including below the floor (the leak this module
    exists to close),
  * a dataset with no positives reports sensitivity None, never 0.0,
  * the shortcut baseline refuses to invent a prior for an unseen dataset.
"""

import numpy as np
import pandas as pd
import pytest

from modeling.metrics import (
    dataset_prior_auroc,
    format_per_dataset,
    metrics_at_threshold,
    per_dataset_metrics,
    prevalence_by_dataset,
    select_threshold_at_floor,
)


# ---------------------------------------------------------------------------
# select_threshold_at_floor
# ---------------------------------------------------------------------------

def test_threshold_meets_floor_with_max_specificity():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.75, 0.4, 0.7, 0.8, 0.9, 0.95])
    thr = select_threshold_at_floor(labels, probs, 0.8)
    # 0.7 catches 4/5 positives (0.80) with one false positive (0.75); any
    # higher threshold drops below the floor.
    assert thr == pytest.approx(0.7)
    m = metrics_at_threshold(labels, probs, thr)
    assert m["sensitivity"] == pytest.approx(0.8)
    assert m["specificity"] == pytest.approx(0.75)


def test_threshold_is_none_for_single_class_split():
    assert select_threshold_at_floor(np.zeros(5), np.linspace(0, 1, 5), 0.8) is None


def test_val_threshold_on_test_can_fail_the_floor():
    # The leak being fixed: a test-tuned threshold always "passes". A
    # val-tuned one must be allowed to report a real failure on test.
    val_labels = np.array([0, 0, 1, 1, 1, 1, 1])
    val_probs = np.array([0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9])
    thr = select_threshold_at_floor(val_labels, val_probs, 0.8)

    test_labels = np.array([0, 0, 1, 1, 1, 1, 1])
    test_probs = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.8, 0.9])  # positives score lower
    m = metrics_at_threshold(test_labels, test_probs, thr)
    assert m["sensitivity"] < 0.8


# ---------------------------------------------------------------------------
# metrics_at_threshold
# ---------------------------------------------------------------------------

def test_metrics_at_threshold_counts():
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.2, 0.8, 0.1])
    m = metrics_at_threshold(labels, probs, 0.5)
    assert (m["n"], m["positives"]) == (4, 2)
    assert m["sensitivity"] == 0.5
    assert m["specificity"] == 0.5
    assert m["precision"] == 0.5
    assert m["worth_f1"] == pytest.approx(0.5)


def test_rates_are_none_not_zero_without_denominator():
    # No positives: sensitivity is undefined. Reporting 0.0 would read as a
    # safety failure that never happened.
    m = metrics_at_threshold(np.array([0, 0]), np.array([0.1, 0.2]), 0.5)
    assert m["sensitivity"] is None
    assert m["precision"] is None
    assert m["worth_f1"] is None
    assert m["specificity"] == 1.0


def test_worth_f1_is_zero_when_positives_all_missed():
    m = metrics_at_threshold(np.array([1, 0]), np.array([0.1, 0.9]), 0.5)
    assert m["sensitivity"] == 0.0
    assert m["worth_f1"] == 0.0


# ---------------------------------------------------------------------------
# Per-dataset metrics + shortcut baseline
# ---------------------------------------------------------------------------

def _pooled_shortcut_fixture():
    """Scores that are PERFECT at telling datasets apart and USELESS within one.

    cbis is mostly positive and scored high; rsna mostly negative and scored
    low. Within each dataset the score is constant, so it carries no signal.
    """
    labels = np.array([1, 1, 1, 0] + [1, 0, 0, 0, 0, 0, 0, 0])
    datasets = np.array(["cbis"] * 4 + ["rsna"] * 8)
    probs = np.where(datasets == "cbis", 0.9, 0.1)
    return labels, probs, datasets


def test_pooled_auroc_can_be_pure_dataset_shortcut():
    from sklearn.metrics import roc_auc_score

    labels, probs, datasets = _pooled_shortcut_fixture()
    assert roc_auc_score(labels, probs) > 0.8  # looks good pooled...
    per = per_dataset_metrics(labels, probs, datasets, threshold=0.5)
    assert per["cbis"]["auroc"] == pytest.approx(0.5)  # ...is chance within each
    assert per["rsna"]["auroc"] == pytest.approx(0.5)


def test_dataset_prior_auroc_matches_the_shortcut():
    labels, probs, datasets = _pooled_shortcut_fixture()
    prior = dataset_prior_auroc(labels, datasets, {"cbis": 0.87, "rsna": 0.02})
    from sklearn.metrics import roc_auc_score
    assert prior == pytest.approx(roc_auc_score(labels, probs))


def test_dataset_prior_auroc_rejects_unseen_dataset():
    labels, _, datasets = _pooled_shortcut_fixture()
    with pytest.raises(ValueError, match="No train prevalence"):
        dataset_prior_auroc(labels, datasets, {"cbis": 0.87})


def test_per_dataset_single_class_dataset_has_none_auroc():
    labels = np.array([1, 0, 0, 0])
    datasets = np.array(["a", "a", "b", "b"])
    per = per_dataset_metrics(labels, np.array([0.9, 0.1, 0.2, 0.3]), datasets, 0.5)
    assert per["b"]["auroc"] is None
    assert per["b"]["sensitivity"] is None
    assert per["b"]["n"] == 2 and per["b"]["positives"] == 0


def test_per_dataset_without_threshold_reports_auroc_only():
    labels, probs, datasets = _pooled_shortcut_fixture()
    per = per_dataset_metrics(labels, probs, datasets, threshold=None)
    assert "sensitivity" not in per["cbis"]
    assert "cbis:auc=0.500" in format_per_dataset(per)


def test_per_dataset_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        per_dataset_metrics(np.array([0, 1]), np.array([0.1]), np.array(["a", "a"]), 0.5)


def test_prevalence_by_dataset():
    df = pd.DataFrame({"dataset": ["a", "a", "b", "b", "b", "b"],
                       "canonical_label": [1, 0, 1, 0, 0, 0]})
    assert prevalence_by_dataset(df, "dataset", "canonical_label") == {
        "a": 0.5, "b": 0.25,
    }
