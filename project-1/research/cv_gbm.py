"""NumPy GBM on all raw columns. Test score = mean of the 5 fold models.

python cv_gbm.py TAG --trees 500 --depth 5 --lr 0.05 [--seed 0] [--bins v1|v2]
v1: raw codes.  v2: 'none' codes -> 0, ALCDAY5 as days/month.
"""

import argparse
import time

import numpy as np

from common import CACHE, folds, load_raw, report, save_scores
from gbm import GBM, apply_bins, make_bins, recode_none_codes
from wide import DROP

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--trees", type=int, default=500)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--row_sub", type=float, default=0.5)
ap.add_argument("--col_sub", type=float, default=0.5)
ap.add_argument("--min_hess", type=float, default=20.0)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bins", default="v1")
ap.add_argument("--drop", nargs="*", default=[], help="raw column names to remove")
a = ap.parse_args()


def alcday_per_month(v):
    out = np.full(v.shape, np.nan)
    out[v == 888] = 0.0
    wk = (v >= 101) & (v <= 199)
    mo = (v >= 201) & (v <= 299)
    out[wk] = (v[wk] - 100) * 30.0 / 7.0
    out[mo] = v[mo] - 200
    return out


t0 = time.time()
bf = CACHE / f"bins_{a.bins}.npz"
if not bf.exists():
    X, XT, y, ids, hdr = load_raw()
    keep = [j for j, c in enumerate(hdr) if c not in DROP]
    A, AT = X[:, keep], XT[:, keep]
    if a.bins == "v2":
        ja = hdr.index("ALCDAY5")
        A = np.column_stack([recode_none_codes(A, A), alcday_per_month(X[:, ja])])
        AT = np.column_stack(
            [recode_none_codes(AT, X[:, keep]), alcday_per_month(XT[:, ja])]
        )
    edges = make_bins(A)
    np.savez(bf, B=apply_bins(A, edges), BT=apply_bins(AT, edges), y=y, ids=ids)
    del X, XT, A, AT
d = np.load(bf)
B, BT, y = d["B"], d["BT"], d["y"]
if a.drop:
    hdr = load_raw()[4]
    cols = [c for c in hdr if c not in DROP] + (
        ["ALCDAY5_month"] if a.bins == "v2" else []
    )
    keep = [j for j, c in enumerate(cols) if c not in set(a.drop)]
    assert len(keep) == B.shape[1] - len(a.drop), "unknown column in --drop"
    B, BT = B[:, keep], BT[:, keep]
print(f"bins {a.bins} {B.shape} {time.time() - t0:.0f}s", flush=True)

fl = folds(y)
cps = sorted({c for c in range(100, a.trees + 1, 100)} | {a.trees})
oof = {c: np.zeros(len(y)) for c in cps}
test = {c: np.zeros(len(BT)) for c in cps}
for i, (tr, va) in enumerate(fl):
    m = GBM(
        a.trees,
        a.depth,
        a.lr,
        lam=a.lam,
        min_hess=a.min_hess,
        row_sub=a.row_sub,
        col_sub=a.col_sub,
        seed=42 + i + 1000 * a.seed,
    )
    m.fit(B[tr], y[tr], B_eval=np.vstack([B[va], BT]), checkpoints=set(cps))
    for c in cps:
        s = m.snapshots[c]
        oof[c][va] = s[: len(va)]
        test[c] += s[len(va) :] / len(fl)
    print(f"  fold {i} {time.time() - t0:.0f}s", flush=True)

for c in cps:
    report(f"gbm_{a.tag} trees={c}", y, oof[c], fl)
    save_scores(f"gbm_{a.tag}_t{c}", oof[c], test[c])
print(f"done {time.time() - t0:.0f}s", flush=True)
