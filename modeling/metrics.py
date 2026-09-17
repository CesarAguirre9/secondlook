# Threshold selection and per-dataset metrics for the Second Look binary head.
#
# Pure numpy/sklearn (no TensorFlow import), so the safety-critical threshold
# logic is unit-testable on CPU in milliseconds.
#
# WHY THIS MODULE EXISTS -- two evaluation leaks in the original protocol:
#
#   1. The deployable operating point was chosen ON THE TEST SET and then scored
#      on that same test set. That guarantees the 0.80 sensitivity floor is met
#      "by construction" and makes op_specificity / op_precision optimistic.
#      The honest order is: pick the threshold on VAL, freeze it, then report
#      whatever sensitivity it actually achieves on TEST -- which can fall below
#      the floor, and that is precisely the safety signal we need to see.
#
#   2. The combined test split pools datasets whose positive rates differ by
#      ~40x (CBIS ~87%, RSNA ~2%) and whose images look different (digitized
#      film vs digital). A model can earn pooled AUROC just by recognizing the
#      SOURCE DATASET. dataset_prior_auroc() measures how much AUROC that
#      shortcut alone is worth; per_dataset_metrics() reports the numbers that
#      cannot be inflated by it.

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _both_classes_present(labels: np.ndarray) -> bool:
    return len(np.unique(labels)) >= 2


def select_threshold_at_floor(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sensitivity_floor: float,
) -> float | None:
    """Highest threshold whose sensitivity on THIS split clears the floor.

    Walks the ROC curve (thresholds descending, sensitivity non-decreasing) and
    takes the first point that meets the floor: the smallest false-positive rate
    among all thresholds that satisfy the sensitivity requirement.

    Call this on the VALIDATION split, then apply the result to test with
    metrics_at_threshold(). Returns None when no threshold reaches the floor or
    the split is single-class.
    """
    labels = np.asarray(labels).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    if not _both_classes_present(labels):
        return None

    # drop_intermediate=False: the default prunes collinear ROC points, which can
    # skip the highest qualifying threshold and return a lower one at the same FPR.
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    meets = tpr >= sensitivity_floor
    if not meets.any():
        return None
    thr = float(thresholds[int(np.argmax(meets))])
    # sklearn prepends an infinite threshold; clamp to 1.0 for a usable value.
    return thr if np.isfinite(thr) else 1.0


def metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Confusion-derived metrics for the WORTH class at a FIXED threshold.

    Rates with an empty denominator are None (not 0): a dataset with no
    positives has no sensitivity, and reporting 0.0 would read as a safety
    failure that did not happen.
    """
    labels = np.asarray(labels).ravel().astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    pred = (probabilities >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())

    def _rate(num: int, den: int) -> float | None:
        return float(num / den) if den else None

    sensitivity = _rate(tp, tp + fn)
    precision = _rate(tp, tp + fp)
    if sensitivity is None or precision is None or (sensitivity + precision) == 0:
        worth_f1 = None if sensitivity is None else 0.0
    else:
        worth_f1 = float(2 * precision * sensitivity / (precision + sensitivity))

    return {
        "threshold": float(threshold),
        "n": int(len(labels)),
        "positives": int(tp + fn),
        "sensitivity": sensitivity,
        "specificity": _rate(tn, tn + fp),
        "precision": precision,
        "worth_f1": worth_f1,
    }


def per_dataset_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    datasets,
    threshold: float | None,
) -> dict[str, dict]:
    """AUROC plus fixed-threshold metrics computed WITHIN each source dataset.

    Within a single dataset the "which dataset is this?" shortcut carries no
    information, so these are the numbers that reflect real discrimination.
    ``threshold`` should be the val-selected deploy threshold; None skips the
    thresholded metrics and reports AUROC only.
    """
    labels = np.asarray(labels).ravel().astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    datasets = np.asarray(datasets).ravel().astype(str)
    if not len(labels) == len(probabilities) == len(datasets):
        raise ValueError(
            f"labels ({len(labels)}), probabilities ({len(probabilities)}) and "
            f"datasets ({len(datasets)}) must be the same length."
        )

    out: dict[str, dict] = {}
    for name in sorted(np.unique(datasets)):
        mask = datasets == name
        y, p = labels[mask], probabilities[mask]
        entry = {
            "n": int(mask.sum()),
            "positives": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)) if _both_classes_present(y) else None,
        }
        if threshold is not None:
            at = metrics_at_threshold(y, p, threshold)
            entry.update({k: at[k] for k in
                          ("sensitivity", "specificity", "precision", "worth_f1")})
        out[str(name)] = entry
    return out


def dataset_prior_auroc(
    labels: np.ndarray,
    datasets,
    train_prevalence: dict[str, float],
) -> float | None:
    """AUROC of a "model" that outputs only its dataset's TRAIN positive rate.

    This is the shortcut baseline: it sees no pixels at all. If the trained
    model's pooled AUROC is not clearly above this, the pooled number is mostly
    measuring dataset identity, not findings.

    Raises:
        ValueError: if a test dataset has no train prevalence (silently scoring
            it as 0 would fabricate a baseline).
    """
    labels = np.asarray(labels).ravel().astype(np.int64)
    datasets = np.asarray(datasets).ravel().astype(str)
    missing = set(np.unique(datasets)) - set(train_prevalence)
    if missing:
        raise ValueError(
            f"No train prevalence for dataset(s) {sorted(missing)}; "
            f"have {sorted(train_prevalence)}."
        )
    if not _both_classes_present(labels):
        return None
    scores = np.array([train_prevalence[d] for d in datasets], dtype=np.float64)
    return float(roc_auc_score(labels, scores))


def prevalence_by_dataset(df, dataset_col: str, label_col: str) -> dict[str, float]:
    """Positive rate per dataset -- feed it the TRAIN split for the prior AUROC."""
    rates = df.groupby(dataset_col)[label_col].mean()
    return {str(k): float(v) for k, v in rates.items()}


def format_per_dataset(per_dataset: dict[str, dict]) -> str:
    """Compact one-line rendering for the sweep summary CSV."""
    def _f(v):
        return "na" if v is None else f"{v:.3f}"

    parts = []
    for name, m in per_dataset.items():
        s = f"{name}:auc={_f(m.get('auroc'))}"
        if "specificity" in m:
            s += (f"/sens={_f(m.get('sensitivity'))}/spec={_f(m.get('specificity'))}"
                  f"/prec={_f(m.get('precision'))}")
        parts.append(s)
    return " | ".join(parts)
