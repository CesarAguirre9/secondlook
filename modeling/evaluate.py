# Evaluation for the Second Look baseline classifier.
#
# Sensitivity (recall) on the WORTH_SECOND_LOOK class is the primary metric.
# A model that misses WORTH cases causes false reassurance — the worst
# failure mode defined in CLAUDE.md.
#
# This evaluator reports the sensitivity-first protocol:
#   1. AUROC — the honest, threshold-INDEPENDENT discrimination metric. On
#      imbalanced data (CBIS is ~87% positive) raw accuracy/sensitivity at a
#      fixed 0.5 threshold are misleading; AUROC is not.
#   2. DEPLOY point — the threshold that maximizes specificity subject to WORTH
#      sensitivity >= WORTH_SENSITIVITY_FLOOR (0.80), SELECTED ON VALIDATION
#      (pass ``deploy_threshold``) and then applied unchanged to test. Test
#      sensitivity at that threshold can land below the floor; that is the
#      honest safety check. Reported with PRECISION and WORTH-F1: on a
#      ~94%-negative screening set, specificity flatters the false-alarm burden
#      and precision does not.
#      The older test-selected "oracle" point (operating_point) is still
#      reported for continuity with past sweeps, but it is optimistic by
#      construction: its threshold was tuned on the labels it is scored on.
#   2b. PER-DATASET metrics + the dataset-prior AUROC shortcut baseline (pass
#      ``dataset_col`` / ``train_prevalence``). Pooled AUROC across CBIS/RSNA/
#      VinDr can be earned by recognizing the source dataset; see
#      modeling.metrics.
#   3. Calibration — Brier score, Expected Calibration Error, and an optional
#      reliability diagram. Needed before confidence can drive the UX tiers.
#
# It still prints per-class sensitivity + confusion at the reference threshold
# and explicitly warns (and optionally raises) if positive-class sensitivity
# falls below the safety floor.
#
# Usage:
#   from modeling.evaluate import evaluate_baseline
#   results = evaluate_baseline(model, test_df, image_dir="data/images/")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from config.constants import INPUT_SIZE
from modeling.baseline_classifier import (
    LABEL_ORDER,
    POSITIVE_CLASS_INDEX,
    WORTH_SENSITIVITY_FLOOR,
)
from modeling.calibrate import CALIBRATION_BINS, calibration_curve
from modeling.metrics import (
    dataset_prior_auroc,
    metrics_at_threshold,
    per_dataset_metrics,
    select_threshold_at_floor,
)
from modeling.train import _build_dataset


def evaluate_baseline(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    image_dir: str,
    image_col: str = "image_path",
    label_col: str = "label",
    input_size: tuple = INPUT_SIZE,
    batch_size: int = 32,
    threshold: float = 0.5,
    raise_on_unsafe: bool = False,
    sensitivity_floor: float = WORTH_SENSITIVITY_FLOOR,
    output_dir: str | None = None,
    deploy_threshold: float | None = None,
    dataset_col: str | None = None,
    train_prevalence: dict[str, float] | None = None,
) -> dict:
    """Evaluate the baseline model on the test set (sensitivity-first protocol).

    Reports AUROC, the operating point that meets the WORTH sensitivity floor
    at maximum specificity, calibration (Brier + ECE), and — at the reference
    ``threshold`` — per-class sensitivity and a 2x2 confusion matrix. Flags
    explicitly if WORTH_SECOND_LOOK sensitivity at ``threshold`` is below the
    safety floor.

    Args:
        model: Trained Keras model (or path string to a saved .keras file).
        test_df: Test split DataFrame.
        image_dir: Root directory containing image files (use "" if image_col
                   already holds absolute paths).
        image_col: Column with image filenames/paths.
        label_col: Column with binary labels (int 0 or 1).
        input_size: Must match the size used during training.
        batch_size: Inference batch size.
        threshold: Reference threshold for the confusion matrix + floor check.
                   NOTE: the deployable operating point is chosen separately
                   from the sensitivity floor (see operating_point in the
                   return value); this fixed threshold is kept for continuity
                   and as a sanity reference.
        raise_on_unsafe: If True, raises RuntimeError when WORTH sensitivity at
                         ``threshold`` is below the floor. Use this in CI.
        sensitivity_floor: Minimum acceptable WORTH sensitivity (default 0.80).
        output_dir: If given, write a reliability-diagram PNG here. Accepts a
                    local path or a gs:// URI (written via tf.io.gfile).
        deploy_threshold: Threshold selected on the VALIDATION split (see
                    modeling.metrics.select_threshold_at_floor). When given, the
                    honest deploy_point metrics are computed on test at exactly
                    this threshold. None skips them (the oracle point remains).
        dataset_col: Column in test_df naming the source dataset. When given,
                    per-dataset metrics are reported.
        train_prevalence: {dataset: TRAIN positive rate}. With dataset_col,
                    reports the dataset-prior AUROC shortcut baseline.

    Returns:
        Dict with keys: per_class_sensitivity, confusion_matrix, report_str,
        worth_sensitivity, passed_safety_floor, threshold, auroc, macro_f1,
        per_class_f1, operating_point (TEST-selected oracle: threshold/
        sensitivity/specificity/precision/worth_f1), deploy_point (VAL-selected
        threshold applied to test, or None), deploy_passed_floor, per_dataset,
        dataset_prior_auroc, brier_score, ece, reliability_diagram_path, and the
        raw probabilities + true_labels arrays.
    """
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)

    test_ds = _build_dataset(
        test_df, image_dir, image_col, label_col, input_size, batch_size, shuffle=False
    )

    true_labels = np.asarray([int(y) for y in test_df[label_col]])
    probabilities = model.predict(test_ds, verbose=1).ravel()
    predicted_labels = (probabilities >= threshold).astype(np.int64)

    per_class_sensitivity = recall_score(
        true_labels, predicted_labels, average=None, labels=[0, 1]
    )
    worth_sensitivity = per_class_sensitivity[POSITIVE_CLASS_INDEX]

    # Macro-F1: unweighted mean of per-class F1. Unlike accuracy (which a
    # majority-class "vote everything positive" collapse can game — a trivial
    # all-positive model scores CBIS accuracy ~0.91), macro-F1 tanks toward 0.5
    # if the model ignores a class. Reported as a headline guard against that.
    macro_f1 = float(
        f1_score(true_labels, predicted_labels, labels=[0, 1],
                 average="macro", zero_division=0)
    )
    per_class_f1 = f1_score(
        true_labels, predicted_labels, labels=[0, 1], average=None, zero_division=0
    )

    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=LABEL_ORDER,
        digits=3,
        zero_division=0,
    )

    # Threshold-independent + operating-point + calibration metrics.
    auroc = _compute_auroc(true_labels, probabilities)
    operating_point = _operating_point_at_floor(
        true_labels, probabilities, sensitivity_floor
    )
    deploy_point = None
    deploy_passed = None
    if deploy_threshold is not None:
        deploy_point = metrics_at_threshold(true_labels, probabilities, deploy_threshold)
        deploy_passed = (
            deploy_point["sensitivity"] is not None
            and deploy_point["sensitivity"] >= sensitivity_floor
        )

    per_dataset = None
    prior_auroc = None
    if dataset_col is not None:
        datasets = test_df[dataset_col].astype(str).to_numpy()
        per_dataset = per_dataset_metrics(
            true_labels, probabilities, datasets, deploy_threshold
        )
        if train_prevalence is not None:
            prior_auroc = dataset_prior_auroc(true_labels, datasets, train_prevalence)
    brier, ece, curve = calibration_curve(
        true_labels, probabilities, n_bins=CALIBRATION_BINS
    )
    diagram_path = None
    if output_dir is not None and curve is not None:
        diagram_path = _save_reliability_diagram(curve, brier, ece, output_dir)

    _print_results(
        per_class_sensitivity, worth_sensitivity, cm, report, threshold,
        auroc, operating_point, brier, ece, sensitivity_floor, macro_f1,
        deploy_point, per_dataset, prior_auroc,
    )

    passed = worth_sensitivity >= sensitivity_floor
    if not passed:
        msg = (
            f"\nSAFETY WARNING: WORTH_SECOND_LOOK sensitivity is "
            f"{worth_sensitivity:.3f}, below the required floor of "
            f"{sensitivity_floor}. This model risks false reassurance "
            f"and must not be deployed."
        )
        print(msg)
        if raise_on_unsafe:
            raise RuntimeError(msg)

    return {
        "per_class_sensitivity": {
            LABEL_ORDER[i]: float(s) for i, s in enumerate(per_class_sensitivity)
        },
        "confusion_matrix": cm,
        "report_str": report,
        "worth_sensitivity": float(worth_sensitivity),
        "passed_safety_floor": bool(passed),
        "threshold": float(threshold),
        "auroc": auroc,
        "macro_f1": macro_f1,
        "per_class_f1": {LABEL_ORDER[i]: float(f) for i, f in enumerate(per_class_f1)},
        "operating_point": operating_point,
        "deploy_point": deploy_point,
        "deploy_passed_floor": deploy_passed,
        "per_dataset": per_dataset,
        "dataset_prior_auroc": prior_auroc,
        "brier_score": brier,
        "ece": ece,
        "reliability_diagram_path": diagram_path,
        # Raw arrays so downstream steps (temperature scaling, error analysis)
        # do not have to re-run inference over the whole test split.
        "probabilities": probabilities,
        "true_labels": true_labels,
    }


# ---------------------------------------------------------------------------
# Threshold-independent + operating-point + calibration metrics
# ---------------------------------------------------------------------------

def _both_classes_present(true_labels: np.ndarray) -> bool:
    """AUROC / ROC-sweep need at least one sample of each class."""
    return len(np.unique(true_labels)) >= 2


def _compute_auroc(true_labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    """AUROC - the honest, threshold-independent metric. None if single-class."""
    if not _both_classes_present(true_labels):
        print(
            "\nNOTE: test set has a single class; AUROC is undefined and "
            "reported as None. (Sensitivity/specificity trade-offs are "
            "meaningless without both classes present.)"
        )
        return None
    return float(roc_auc_score(true_labels, probabilities))


def _operating_point_at_floor(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    sensitivity_floor: float,
) -> dict | None:
    """TEST-selected ("oracle") threshold with max specificity at the floor.

    OPTIMISTIC BY CONSTRUCTION: the threshold is tuned on the same labels it is
    scored on, so the floor is always met. Kept only so new runs stay comparable
    with earlier sweep summaries -- read deploy_point for the honest number.
    Returns None if no threshold meets the floor (or single-class).
    """
    thr = select_threshold_at_floor(true_labels, probabilities, sensitivity_floor)
    if thr is None:
        if _both_classes_present(true_labels):
            print(
                f"\nNOTE: no threshold reaches the {sensitivity_floor:.2f} "
                f"sensitivity floor on this test set; no safe operating point exists."
            )
        return None
    m = metrics_at_threshold(true_labels, probabilities, thr)
    return {k: m[k] for k in
            ("threshold", "sensitivity", "specificity", "precision", "worth_f1")}


def _save_reliability_diagram(
    curve: dict, brier: float, ece: float, output_dir: str
) -> str | None:
    """Write a reliability-diagram PNG to output_dir (local or gs://)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — no display on training VMs
        import matplotlib.pyplot as plt
    except ImportError:
        print("NOTE: matplotlib unavailable; skipping reliability diagram.")
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(curve["mean_predicted"], curve["observed_rate"], "o-", label="model")
    ax.set_xlabel("Mean predicted P(WORTH_SECOND_LOOK)")
    ax.set_ylabel("Observed WORTH rate")
    ax.set_title(f"Reliability diagram (Brier={brier:.3f}, ECE={ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()

    dest = output_dir.rstrip("/") + "/reliability_diagram.png"
    if "://" in output_dir:
        # Write locally then copy to the remote filesystem via tf.io.gfile.
        import os
        import tempfile
        tf.io.gfile.makedirs(output_dir)
        tmp = os.path.join(tempfile.gettempdir(), "reliability_diagram.png")
        fig.savefig(tmp, dpi=120)
        plt.close(fig)
        tf.io.gfile.copy(tmp, dest, overwrite=True)
    else:
        tf.io.gfile.makedirs(output_dir)
        fig.savefig(dest, dpi=120)
        plt.close(fig)
    print(f"Reliability diagram written to: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_results(
    per_class_sensitivity: np.ndarray,
    worth_sensitivity: float,
    cm: np.ndarray,
    report: str,
    threshold: float,
    auroc: float | None,
    operating_point: dict | None,
    brier: float,
    ece: float,
    sensitivity_floor: float,
    macro_f1: float,
    deploy_point: dict | None = None,
    per_dataset: dict | None = None,
    prior_auroc: float | None = None,
) -> None:
    print("\n" + "=" * 60)
    print("SECOND LOOK - BASELINE EVALUATION")
    print("=" * 60)

    # Threshold-independent discrimination first — the honest headline number.
    auroc_str = f"{auroc:.3f}" if auroc is not None else "N/A (single-class test set)"
    print(f"\nAUROC (threshold-independent): {auroc_str}")
    # Imbalance-robust headline alongside AUROC. Guards against a majority-class
    # collapse that accuracy would otherwise flatter.
    print(f"Macro-F1 (imbalance-robust)  : {macro_f1:.3f}")
    if prior_auroc is not None:
        # The shortcut baseline: AUROC from dataset identity alone, no pixels.
        print(f"Dataset-prior AUROC (no-pixel shortcut baseline): {prior_auroc:.3f}")

    def _fmt(v):
        return "n/a" if v is None else f"{v:.3f}"

    print("\nDEPLOY point (threshold selected on VAL, applied to test):")
    if deploy_point is None:
        print("  not computed - no val-selected threshold was supplied.")
    else:
        sens = deploy_point["sensitivity"]
        verdict = "PASS" if sens is not None and sens >= sensitivity_floor else "FAIL"
        print(f"  threshold  : {deploy_point['threshold']:.3f}")
        print(f"  sensitivity: {_fmt(sens)} (WORTH; floor {sensitivity_floor:.2f}) "
              f"-> {verdict}")
        print(f"  specificity: {_fmt(deploy_point['specificity'])}")
        print(f"  precision  : {_fmt(deploy_point['precision'])}")
        print(f"  WORTH F1   : {_fmt(deploy_point['worth_f1'])}")

    if per_dataset:
        print("\nPer-dataset (dataset identity cannot inflate these):")
        print(f"  {'dataset':10s} {'n':>7s} {'pos':>6s} {'AUROC':>6s} "
              f"{'sens':>6s} {'spec':>6s} {'prec':>6s}")
        for name, m in per_dataset.items():
            print(f"  {name:10s} {m['n']:>7d} {m['positives']:>6d} "
                  f"{_fmt(m['auroc']):>6s} {_fmt(m.get('sensitivity')):>6s} "
                  f"{_fmt(m.get('specificity')):>6s} {_fmt(m.get('precision')):>6s}")

    print("\nORACLE operating point (threshold tuned ON TEST - optimistic, "
          "kept for comparison with earlier sweeps):")
    if operating_point is None:
        # ASCII only: this repo has already been bitten by a UnicodeEncodeError
        # on Windows cp1252 consoles. Keep every print in this file ASCII.
        print(f"  NONE - no threshold meets the {sensitivity_floor:.2f} "
              f"sensitivity floor on this test set.")
    else:
        print(f"  threshold  : {operating_point['threshold']:.3f}")
        print(f"  sensitivity: {operating_point['sensitivity']:.3f} "
              f"(WORTH; floor {sensitivity_floor:.2f})")
        print(f"  specificity: {operating_point['specificity']:.3f}")
        print(f"  precision  : {operating_point['precision']:.3f} "
              f"(share of flagged cases that really are WORTH)")
        print(f"  WORTH F1   : {operating_point['worth_f1']:.3f}")

    print("\nCalibration:")
    print(f"  Brier score: {brier:.3f}  (lower is better)")
    print(f"  ECE        : {ece:.3f}  (expected calibration error, lower better)")

    print("\n" + "-" * 60)
    print(f"Reference threshold: {threshold:.2f}")

    print("\nSensitivity (Recall) per class:")
    for i, name in enumerate(LABEL_ORDER):
        marker = " <-- PRIMARY METRIC" if i == POSITIVE_CLASS_INDEX else ""
        print(f"  {name:25s}: {per_class_sensitivity[i]:.3f}{marker}")

    status = "PASS" if worth_sensitivity >= sensitivity_floor else f"FAIL (floor: {sensitivity_floor})"
    print(f"\nWORTH_SECOND_LOOK sensitivity floor check: {status}")

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = f"{'':25s}" + "".join(f"{n:>25s}" for n in LABEL_ORDER)
    print(header)
    for i, name in enumerate(LABEL_ORDER):
        row = f"{name:25s}" + "".join(f"{cm[i, j]:>25d}" for j in range(len(LABEL_ORDER)))
        print(row)

    print("\nFull Classification Report:")
    print(report)
    print("=" * 60)
    print(
        "NOTE: Accuracy is not the primary metric here. On imbalanced data, "
        "read AUROC + the operating point at the sensitivity floor first. A "
        "model that catches WORTH cases at the cost of some false alarms is "
        "preferable to one that misses WORTH cases."
    )
