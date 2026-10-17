"""Reference: frozen base pipeline (176 cols), IRLS logistic. Saves OOF and test scores."""

import time

import numpy as np

from common import DATA, folds, report, save_scores
from clinical_fe import ClinicalFE
from interactions import TopKPairInteractions, _logreg_irls, _sigmoid
from preprocess import build_train_test

t0 = time.time()
tx, y, te, pp = build_train_test(
    DATA / "x_train.csv", DATA / "y_train.csv", DATA / "x_test.csv"
)
cfe = ClinicalFE(add_bias=False)
X = np.hstack([tx, cfe.fit_transform(DATA / "x_train.csv")])
XT = np.hstack([te, cfe.transform(DATA / "x_test.csv")])
print(f"matrices {X.shape} {time.time() - t0:.0f}s", flush=True)

fl = folds(y)
oof = np.zeros(len(y))
for i, (tr, va) in enumerate(fl):
    inter = TopKPairInteractions(k=12, lam=1.0, bias_idx=0, scale=True)
    A = inter.fit_transform(X[tr], y[tr])
    w = _logreg_irls(A, y[tr], lam=1.0, bias_idx=0)
    oof[va] = _sigmoid(inter.transform(X[va]) @ w)
report("logistic", y, oof, fl)

inter = TopKPairInteractions(k=12, lam=1.0, bias_idx=0, scale=True)
A = inter.fit_transform(X, y)
w = _logreg_irls(A, y, lam=1.0, bias_idx=0)
test = _sigmoid(inter.transform(XT) @ w)
save_scores("logistic", oof, test)
print(f"done {time.time() - t0:.0f}s", flush=True)
