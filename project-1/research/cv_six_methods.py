"""The six course methods on the frozen base matrix (176 cols), same 5-fold protocol.

Interactions are refitted inside each fold. Every method outputs a score; the F1
threshold is chosen as everywhere else (pooled and nested).
"""

import sys
import time

import numpy as np

from common import DATA, folds, report
from clinical_fe import ClinicalFE
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)
from interactions import TopKPairInteractions
from preprocess import build_train_test

t0 = time.time()
tx, y, _, _ = build_train_test(
    DATA / "x_train.csv", DATA / "y_train.csv", DATA / "x_test.csv"
)
X = np.hstack([tx, ClinicalFE(add_bias=False).fit_transform(DATA / "x_train.csv")])
fl = folds(y)
splits = []
for tr, va in fl:
    inter = TopKPairInteractions(k=12, lam=1.0, bias_idx=0, scale=True)
    splits.append((tr, va, inter.fit_transform(X[tr], y[tr]), inter.transform(X[va])))
print(f"matrices ready {time.time() - t0:.0f}s", flush=True)

np.random.seed(42)
methods = {
    "least_squares": lambda A, b: least_squares(b, A),
    "ridge lam=1e-6": lambda A, b: ridge_regression(b, A, 1e-6),
    "ridge lam=1e-4": lambda A, b: ridge_regression(b, A, 1e-4),
    "ridge lam=1e-2": lambda A, b: ridge_regression(b, A, 1e-2),
    "mse_gd g=0.02 it=1000": lambda A, b: mean_squared_error_gd(
        b, A, np.zeros(A.shape[1]), 1000, 0.02
    ),
    "mse_sgd g=1e-3 it=300k": lambda A, b: mean_squared_error_sgd(
        b, A, np.zeros(A.shape[1]), 300_000, 1e-3
    ),
    "mse_sgd g=1e-4 it=300k": lambda A, b: mean_squared_error_sgd(
        b, A, np.zeros(A.shape[1]), 300_000, 1e-4
    ),
    "mse_sgd g=2e-5 it=600k": lambda A, b: mean_squared_error_sgd(
        b, A, np.zeros(A.shape[1]), 600_000, 2e-5
    ),
    "logistic g=0.5 it=3000": lambda A, b: logistic_regression(
        b, A, np.zeros(A.shape[1]), 3000, 0.5
    ),
    "reg_logistic lam=1e-4 g=0.5 it=3000": lambda A, b: reg_logistic_regression(
        b, A, 1e-4, np.zeros(A.shape[1]), 3000, 0.5
    ),
}

only = sys.argv[1:]
for name, fit in methods.items():
    if only and not any(s in name for s in only):
        continue
    oof = np.zeros(len(y))
    ok = True
    for tr, va, A, V in splits:
        try:
            w, _ = fit(A, y[tr])
        except np.linalg.LinAlgError as e:
            ok = f"LinAlgError: {e}"
            break
        if not np.all(np.isfinite(w)):
            ok = "diverged"
            break
        oof[va] = V @ w
    if ok is not True:
        print(f"[{name}] {ok}", flush=True)
        continue
    report(name, y, oof, fl)
    print(f"    {time.time() - t0:.0f}s", flush=True)
print(f"done {time.time() - t0:.0f}s", flush=True)
