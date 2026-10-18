"""Equal-weight average of GBM logits (no weight tuning), F1 threshold from OOF, submission.

python bag_equal.py OUT_NAME score1 [score2 ...]
"""

import sys

import numpy as np

from common import CACHE, folds, report, write_submission

out, names = sys.argv[1], sys.argv[2:]
d = np.load(CACHE / "bins_v2.npz")
y, ids = d["y"], d["ids"]
fl = folds(y)

oof = np.zeros(len(y))
test = np.zeros(len(ids))
for n in names:
    s = np.load(CACHE / f"scores_{n}.npz")
    oof += s["oof"] / len(names)
    test += s["test"] / len(names)

thr, f_pool, f_nest, auc = report(out, y, oof, fl)
write_submission(
    out, ids, test >= thr, f"oof F1 {f_pool:.4f} nested {f_nest:.4f} AUC {auc:.4f}"
)
