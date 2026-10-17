"""Blend saved scores, apply the HAREHAB1 rule, write submissions.

python blend_weights.py OUT_NAME score1 [score2 ...]
Weights: simplex grid (step 0.1) on standardised logits, chosen on pooled OOF F1.
"""

import itertools
import sys

import numpy as np

from common import best_thr, f1, folds, load_raw, load_scores, write_submission

out = sys.argv[1]
names = sys.argv[2:]
X, XT, y, ids, hdr = load_raw()
fl = folds(y)
harehab = np.isfinite(X[:, hdr.index("HAREHAB1")])
harehab_t = np.isfinite(XT[:, hdr.index("HAREHAB1")])


def to_logit(s):
    if s.min() >= 0 and s.max() <= 1:
        s = np.log(np.clip(s, 1e-7, 1 - 1e-7) / np.clip(1 - s, 1e-7, 1))
    return s


O, T = [], []
for n in names:
    o, t = load_scores(n)
    o, t = to_logit(o), to_logit(t)
    mu, sd = o.mean(), o.std()
    O.append((o - mu) / sd)
    T.append((t - mu) / sd)
O, T = np.array(O), np.array(T)

if len(names) == 1:
    grid = [np.array([1.0])]
else:
    steps = np.arange(0, 11) / 10
    grid = [
        np.array(c)
        for c in itertools.product(steps, repeat=len(names))
        if abs(sum(c) - 1) < 1e-9
    ]

best = None
for wv in grid:
    s = wv @ O
    thr, f = best_thr(y, s)
    if best is None or f > best[1]:
        best = (wv, f, thr)
wv, f_pool, thr = best
s_oof, s_test = wv @ O, wv @ T

nested = np.zeros(len(y), dtype=bool)
for tr, va in fl:
    bw, bf = None, -1
    for g in grid:
        _, fg = best_thr(y[tr], g @ O[:, tr])
        if fg > bf:
            bw, bf = g, fg
    t_tr, _ = best_thr(y[tr], bw @ O[:, tr])
    nested[va] = (bw @ O[:, va]) >= t_tr
f_nest = f1(y, nested)

pred_oof = s_oof >= thr
f_rule = f1(y, pred_oof | harehab)
nested_rule = f1(y, nested | harehab)
print(f"[{out}] weights {dict(zip(names, np.round(wv, 2)))}")
print(
    f"  F1 pooled {f_pool:.4f}  nested {f_nest:.4f}  | + HAREHAB1 rule: pooled {f_rule:.4f} nested {nested_rule:.4f}"
)
print(
    f"  HAREHAB1 rows in train {harehab.sum()} already predicted positive {np.mean(pred_oof[harehab]):.3f}; in test {harehab_t.sum()}"
)

pred_test = s_test >= thr
write_submission(f"{out}", ids, pred_test, f"oof F1 {f_pool:.4f} nested {f_nest:.4f}")
write_submission(
    f"{out}_harehab",
    ids,
    pred_test | harehab_t,
    f"oof F1 {f_rule:.4f} nested {nested_rule:.4f}",
)
