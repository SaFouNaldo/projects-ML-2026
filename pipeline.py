"""Frozen pipeline: base preprocessing + NO_DUP_EXTRAS + top-12 interactions.

Usage
-----
from pipeline import build_matrices
tx, y, te, meta = build_matrices(data_dir)
# tx / te: matrices ready for logistic_regression
# recommended GD hyperparameters (holdout): gamma=0.5, max_iters=3000..5000
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clinical_fe import ClinicalFE, NO_DUP_EXTRAS
from interactions import TopKPairInteractions
from preprocess import Preprocessor, build_train_test

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "notebooks" / "feature_selection" / "selected_features.txt"

# Frozen after OOF IRLS (F1 +0.0036 vs base + NO_DUP alone)
TOP_K_INTERACTIONS = 12


def build_matrices(
    data_dir: Path,
    features_path: Path = FEATURES_PATH,
    top_k: int = TOP_K_INTERACTIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns (X_train, y, X_test, meta).

    X = [base preprocessing | 7 NO_DUP extra features | pairwise top-K]
    The bias column is already included; no second bias is added.
    """
    data_dir = Path(data_dir)
    tx, y, te, pp = build_train_test(
        data_dir / "x_train.csv",
        data_dir / "y_train.csv",
        data_dir / "x_test.csv",
        features_path=features_path,
    )
    cfe = ClinicalFE(add_bias=False)
    z_tr = cfe.fit_transform(data_dir / "x_train.csv")
    z_te = cfe.transform(data_dir / "x_test.csv")
    Xtr = np.hstack([tx, z_tr])
    Xte = np.hstack([te, z_te])

    inter = TopKPairInteractions(k=top_k, bias_idx=0, scale=True)
    Xtr = inter.fit_transform(Xtr, y)
    Xte = inter.transform(Xte)

    meta = {
        "base_names": getattr(pp, "out_names_", None),
        "extra_names": list(cfe.feat_names_),
        "inter_names": list(inter.names_),
        "top_idx": None if inter.top_idx_ is None else inter.top_idx_.tolist(),
        "n_base": tx.shape[1],
        "n_extras": z_tr.shape[1],
        "n_inter": len(inter.pairs_),
        "n_total": Xtr.shape[1],
        "extras": list(NO_DUP_EXTRAS),
        "top_k": top_k,
        "preprocessor": pp,
        "clinical_fe": cfe,
        "interactions": inter,
    }
    return Xtr, y, Xte, meta


__all__ = [
    "build_matrices",
    "FEATURES_PATH",
    "NO_DUP_EXTRAS",
    "TOP_K_INTERACTIONS",
    "Preprocessor",
    "TopKPairInteractions",
]
