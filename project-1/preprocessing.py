"""Data loading, cleaning, encoding and scaling.

Owner: all three of us -- each person fills in the rules for their third of the
columns (see notebooks/feature_summary.csv) but the functions below are shared.

The contract with the rest of the project:
    load_data(...)                  -> x_train, x_test, y_train, train_ids, test_ids
    build_features(x_train, x_test) -> tx_train, tx_test   (bias column included)
Nothing here may look at y_test, and every statistic (means, stds, category
lists) must be computed on the TRAIN part only, then applied to the test part.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_data(data_dir="data/dataset", cache_dir="data/cache"):
    """Load the CSVs (empty cells become NaN) and cache them as .npy files.

    Returns:
        x_train (N, D), x_test (M, D), y_train (N,) in {-1, 1},
        feature_names (list of D strings), train_ids, test_ids
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Cleaning: BRFSS special codes
# ---------------------------------------------------------------------------

# Filled in from the BRFSS 2015 codebook, one entry per column that needs it:
#     "GENHLTH": {"nan": [7, 9], "zero": []}
# "nan" -> values meaning "don't know / refused", "zero" -> values meaning "none".
CODEBOOK_RULES = {}


def clean_special_codes(x, feature_names):
    """Replace the survey's non-answer codes by NaN and the "none" codes by 0.

    Uses CODEBOOK_RULES when the column is listed, and a fallback heuristic
    based on the column's maximum value otherwise.

    Returns:
        A cleaned copy of x, same shape.
    """
    raise NotImplementedError


def drop_columns(x, feature_names, to_drop):
    """Remove the listed columns (near-duplicates, mostly empty, identifiers).

    Returns:
        x without those columns, and the remaining feature names.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def missing_indicators(x, feature_names, blocks):
    """Build one 0/1 "was answered" column per question block.

    Some blocks (cardiac rehab, diabetes, cognitive decline) are only asked to
    people who already have the condition, so the fact that they were answered
    is informative on its own.

    Args:
        blocks: dict block_name -> list of column names belonging to it

    Returns:
        np.ndarray of shape (N, len(blocks))
    """
    raise NotImplementedError


def impute(x_train, x_test, strategy="median"):
    """Fill NaNs using statistics computed on x_train only.

    Args:
        strategy: "median", "mean" or "zero"

    Returns:
        x_train, x_test without NaNs.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Encoding and scaling
# ---------------------------------------------------------------------------


def one_hot(x_train, x_test, feature_names, nominal_columns):
    """One-hot encode the nominal (unordered) categorical columns.

    Categories are taken from the train part; a category seen only in the test
    part maps to an all-zero row for that feature.

    Returns:
        x_train, x_test with the nominal columns replaced by their dummies,
        and the new feature names.
    """
    raise NotImplementedError


def standardize(x_train, x_test):
    """Z-score both sets using the TRAIN mean and standard deviation.

    Constant columns (std == 0) are left untouched instead of dividing by zero.
    """
    raise NotImplementedError


def add_bias(x):
    """Prepend a column of ones. Do this AFTER standardizing."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Feature engineering (owner: person A)
# ---------------------------------------------------------------------------


def build_polynomial(x, degree):
    """Append the powers 2..degree of the continuous columns."""
    raise NotImplementedError


def build_interactions(x, feature_names, pairs):
    """Append the products of the given pairs of columns, e.g. age x GENHLTH."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# The single entry point used by run.py and by the cross-validation
# ---------------------------------------------------------------------------


def build_features(x_train, x_test, feature_names, config=None):
    """Run the whole pipeline: clean -> drop -> flags -> impute -> encode ->
    scale -> engineer -> add bias.

    Args:
        config: dict of switches so the ablation study can turn each step on
            and off, e.g. {"one_hot": True, "degree": 2, "missing_flags": True}

    Returns:
        tx_train (N, D'), tx_test (M, D'), final feature names.
    """
    raise NotImplementedError
