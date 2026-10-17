"""Shared tools for the research experiments: data cache, folds, F1, threshold, submissions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "dataset_unzipped" / "dataset"
CACHE = ROOT / "data" / "cache" / "research"
SUBS = ROOT / "submissions" / "research"
CACHE.mkdir(parents=True, exist_ok=True)
SUBS.mkdir(parents=True, exist_ok=True)

SEED = 42
K = 5


def load_raw():
    """Raw float matrices (NaN for blanks), header without Id, labels in {0,1}, test ids."""
    f = CACHE / "raw.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        return d["X"], d["XT"], d["y"], d["ids"], list(d["hdr"])
    hdr = (DATA / "x_train.csv").open().readline().strip().split(",")
    X = np.genfromtxt(DATA / "x_train.csv", delimiter=",", skip_header=1)
    XT = np.genfromtxt(DATA / "x_test.csv", delimiter=",", skip_header=1)
    y = (
        np.genfromtxt(DATA / "y_train.csv", delimiter=",", skip_header=1)[:, 1] == 1
    ).astype(np.float64)
    ids = XT[:, 0].astype(np.int64)
    X, XT, hdr = X[:, 1:], XT[:, 1:], hdr[1:]
    np.savez(f, X=X, XT=XT, y=y, ids=ids, hdr=np.array(hdr))
    return X, XT, y, ids, hdr


def folds(y, k=K, seed=SEED):
    rng = np.random.default_rng(seed)
    parts = []
    for label in (0.0, 1.0):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        parts.append(np.array_split(idx, k))
    va = [np.concatenate([parts[0][i], parts[1][i]]) for i in range(k)]
    return [
        (np.concatenate([va[j] for j in range(k) if j != i]), va[i]) for i in range(k)
    ]


def f1(y, pred):
    yt = y == 1
    tp = np.sum(pred & yt)
    fp = np.sum(pred & ~yt)
    fn = np.sum(~pred & yt)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def best_thr(y, s):
    """Threshold maximising F1, computed in one sort."""
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(s) + 1)
    f = 2 * tp / (k + ys.sum())
    i = int(np.argmax(f))
    return float(s[order][i]), float(f[i])


def report(name, y, oof, fl):
    thr, f_pool = best_thr(y, oof)
    nested = np.zeros(len(y), dtype=bool)
    for tr, va in fl:
        t, _ = best_thr(y[tr], oof[tr])
        nested[va] = oof[va] >= t
    f_nest = f1(y, nested)
    per_fold = [f1(y[va], oof[va] >= thr) for _, va in fl]
    auc = auc_score(y, oof)
    print(
        f"[{name}] F1 pooled {f_pool:.4f}  nested {f_nest:.4f}  "
        f"folds {np.mean(per_fold):.4f}±{np.std(per_fold):.4f}  AUC {auc:.4f}  thr {thr:.4f}",
        flush=True,
    )
    return thr, f_pool, f_nest, auc


def auc_score(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum()
    nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def save_scores(name, oof, test):
    np.savez(CACHE / f"scores_{name}.npz", oof=oof, test=test)


def load_scores(name):
    d = np.load(CACHE / f"scores_{name}.npz")
    return d["oof"], d["test"]


def write_submission(name, ids, test_pred_bool, note=""):
    pred = np.where(test_pred_bool, 1, -1)
    path = SUBS / f"{name}.csv"
    with open(path, "w") as fh:
        fh.write("Id,Prediction\n")
        for i, p in zip(ids, pred):
            fh.write(f"{int(i)},{int(p)}\n")
    print(
        f"wrote {path.name}  positives {pred.__eq__(1).mean() * 100:.2f}%  {note}",
        flush=True,
    )
    return path


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))
