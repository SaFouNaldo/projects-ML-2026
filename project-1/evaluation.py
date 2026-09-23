"""Metrics, cross-validation, threshold and hyperparameter search.

Owner: person C (with A on the folds and B on the metrics).
Everything here works with labels in {-1, 1}; convert to {0, 1} only inside the
logistic methods.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def confusion_counts(y_true, y_pred):
    """Return (tp, fp, tn, fn) for labels in {-1, 1}."""
    raise NotImplementedError


def accuracy(y_true, y_pred):
    """Fraction of correct predictions. Careful: 91.2% is the trivial baseline."""
    raise NotImplementedError


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive class (+1)."""
    raise NotImplementedError


def fbeta(y_true, y_pred, beta=1.0):
    """F-beta score. beta > 1 puts more weight on recall (fewer missed cases)."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def build_k_indices(y, k_fold, seed=42):
    """Return an array of shape (k_fold, N // k_fold) of shuffled row indices.

    Use the same seed everywhere so that all three of us compare numbers that
    were computed on exactly the same folds.
    """
    raise NotImplementedError


def stratified_k_indices(y, k_fold, seed=42):
    """Same as build_k_indices but keeps the 8.8% positive rate in every fold."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


def best_threshold(y_true, scores, metric="f1", n_grid=101):
    """Scan thresholds over the score range and return (threshold, best_value).

    Must be called on a VALIDATION fold, never on the fold used for training.
    """
    raise NotImplementedError


def predict_labels(scores, threshold):
    """Turn model scores into labels in {-1, 1} using the given threshold."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(y, tx, fit_fn, k_fold=5, seed=42, tune_threshold=True):
    """Run k-fold cross-validation for any model.

    Args:
        fit_fn: callable (y_tr, tx_tr) -> (w, score_fn) where score_fn(tx) gives
            a real-valued score per row (probability or raw prediction)
        tune_threshold: if True, pick the best-F1 threshold on each validation
            fold instead of using a fixed one

    Returns:
        dict with the mean and standard deviation of f1, precision, recall,
        accuracy and of the chosen thresholds.
    """
    raise NotImplementedError


def grid_search(y, tx, fit_factory, param_grid, k_fold=5):
    """Cross-validate every combination in param_grid.

    Args:
        fit_factory: callable(**params) -> fit_fn, as expected by cross_validate
        param_grid: dict param name -> list of values

    Returns:
        A list of (params, results) sorted by decreasing mean F1.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------


def run_ablation(y, x_train, feature_names, configs, out_csv="results/ablation.csv"):
    """Evaluate one preprocessing/model configuration per row and log the table.

    One line per configuration: name, F1, precision, recall, accuracy, threshold,
    training time. This table is the backbone of the results section.
    """
    raise NotImplementedError
