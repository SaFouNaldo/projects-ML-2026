"""Wide logistic on all raw columns. Usage: python cv_wide.py [lam ...]"""

import sys
import time

import numpy as np

from common import CACHE, folds, load_raw, report, save_scores
from implementations import sigmoid
from wide import WideEncoder, adam_logistic

t0 = time.time()
X, XT, y, ids, hdr = load_raw()
zf = CACHE / "wide.npz"
if zf.exists():
    d = np.load(zf)
    Z, ZT = d["Z"], d["ZT"]
else:
    enc = WideEncoder().fit(X, hdr)
    Z, ZT = enc.transform(X), enc.transform(XT)
    np.savez(zf, Z=Z, ZT=ZT)
    (CACHE / "wide_names.txt").write_text("\n".join(enc.names))
print(f"wide {Z.shape}  {time.time() - t0:.0f}s", flush=True)

fl = folds(y)
lams = [float(a) for a in sys.argv[1:]] or [1e-4]
ITERS = 500
for lam in lams:
    oof = np.zeros(len(y))
    for i, (tr, va) in enumerate(fl):
        w, b = adam_logistic(Z[tr], y[tr], lam=lam, iters=ITERS)
        oof[va] = sigmoid(Z[va] @ w + b)
        print(f"  lam {lam:g} fold {i} {time.time() - t0:.0f}s", flush=True)
    report(f"wide lam={lam:g}", y, oof, fl)
    w, b = adam_logistic(Z, y, lam=lam, iters=ITERS)
    save_scores(f"wide_l{lam:g}", oof, sigmoid(ZT @ w + b))
print(f"done {time.time() - t0:.0f}s", flush=True)
