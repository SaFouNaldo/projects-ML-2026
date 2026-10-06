"""Pipeline figé : Sara preprocess + NO_DUP_EXTRAS.

Usage
-----
from pipeline_sara import build_matrices
tx, y, te, meta = build_matrices(data_dir)
# tx / te : matrices prêtes pour logistic_regression
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clinical_fe import ClinicalFE, NO_DUP_EXTRAS
from preprocess_sara import Preprocessor, build_train_test

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "notebooks" / "research_sara" / "features_sara.txt"


def build_matrices(
    data_dir: Path,
    features_path: Path = FEATURES_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Retourne (X_train, y, X_test, meta).

    X = [Sara preprocess | 7 extras NO_DUP] — biais Sara déjà inclus, pas de
    second biais clinique.
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
    meta = {
        "sara_names": getattr(pp, "out_names_", None),
        "extra_names": list(cfe.feat_names_),
        "n_sara": tx.shape[1],
        "n_extras": z_tr.shape[1],
        "n_total": Xtr.shape[1],
        "extras": list(NO_DUP_EXTRAS),
        "preprocessor": pp,
        "clinical_fe": cfe,
    }
    return Xtr, y, Xte, meta


__all__ = ["build_matrices", "FEATURES_PATH", "NO_DUP_EXTRAS", "Preprocessor"]
