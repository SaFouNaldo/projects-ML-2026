"""NumPy GBM on all raw columns. Test score = mean of the 5 fold models.

python cv_gbm.py TAG --trees 500 --depth 5 --lr 0.05 [--seed 0] [--bins v1|v2]
v1: raw codes.  v2: 'none' codes -> 0, ALCDAY5 as days/month.
"""

import argparse
import time

import numpy as np

from common import folds, load_bins, load_raw, report, save_scores
from gbm import GBM
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


t0 = time.time()
B, BT, y, _ = load_bins(a.bins)
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
