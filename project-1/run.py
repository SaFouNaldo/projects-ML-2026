"""Reproduce our best AICrowd submission from scratch.

Usage:
    python run.py

It must regenerate exactly the predictions we submitted: same seed, same
configuration, same output file. Keep it short -- the logic lives in
preprocessing.py, implementations.py and models_extra.py.
"""

import numpy as np

import evaluation
import implementations
import preprocessing
from helpers import create_csv_submission

SEED = 42
DATA_DIR = "data/dataset"
OUTPUT = "submissions/best_submission.csv"

# The winning configuration, copied from results/ablation.csv.
CONFIG = {
    "missing_flags": True,
    "drop_missing_above": 0.9,
    "one_hot": True,
    "degree": 1,
    "impute": "median",
}
HYPERPARAMS = {
    "lambda_": None,
    "gamma": None,
    "max_iters": None,
    "pos_weight": None,
    "threshold": None,
}


def main():
    """Load, preprocess, train, predict and write the submission file."""
    np.random.seed(SEED)
    raise NotImplementedError


if __name__ == "__main__":
    main()
