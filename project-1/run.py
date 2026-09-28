"""run.py — prepare the data, train the model, print the metrics, write the submission.

    python run.py
    python run.py --lambda 1e-4 --gamma 0.3 --iters 1000 --tag test2

Reads what 02_cleaning_data.ipynb produced (data/cache/cleaned.npz) and writes one
submission per run into submissions/, plus a line in submissions/runs.csv.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from implementations import reg_logistic_regression, sigmoid

# ------------------------------------------------------------------ parameters
ap = argparse.ArgumentParser()
ap.add_argument("--lambda", dest="lambda_", type=float, default=1e-3)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--iters", type=int, default=800)
ap.add_argument("--val", type=float, default=0.2, help="validation share")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tag", default="run")
args = ap.parse_args()

t0 = time.time()
SUBS = ROOT / "submissions"
SUBS.mkdir(exist_ok=True)


# =============================================================== 1. PREPARE DATA
d = np.load(ROOT / "data" / "cache" / "cleaned.npz", allow_pickle=False)
X, X_test = d["X"].astype(np.float64), d["XT"].astype(np.float64)
y, ids_test = d["y"].astype(np.float64), d["ids_te"]
print(f"train {X.shape}   test {X_test.shape}   positives {100 * y.mean():.2f} %")

# train / validation split, keeping the same share of positives in both
rng = np.random.default_rng(args.seed)
pos = rng.permutation(np.flatnonzero(y == 1))
neg = rng.permutation(np.flatnonzero(y == 0))
n_val_pos, n_val_neg = int(args.val * len(pos)), int(args.val * len(neg))
val_idx = np.concatenate([pos[:n_val_pos], neg[:n_val_neg]])
tr_idx = np.concatenate([pos[n_val_pos:], neg[n_val_neg:]])

# standardisation: mean and std computed on the TRAINING rows only, then applied
# everywhere. Using the validation or test rows here would be leakage.
mean = X[tr_idx].mean(axis=0)
std = X[tr_idx].std(axis=0)
std[std == 0] = 1.0


def prepare(M):
    """Standardise and add the column of ones for the intercept."""
    Z = (M - mean) / std
    return np.c_[np.ones(len(Z)), Z]


tx_tr, y_tr = prepare(X[tr_idx]), y[tr_idx]
# ---- optional extra features: squares and products of the strongest columns ----
TOP_K = 20          # how many columns to expand (0 = plain model)

if TOP_K:
    corr = np.array([abs(np.corrcoef(X[tr_idx, j], y_tr)[0, 1])
                     if X[tr_idx, j].std() > 0 else 0.0 for j in range(X.shape[1])])
    top = np.argsort(-np.nan_to_num(corr))[:TOP_K]
    pairs = [(a, b) for i, a in enumerate(top) for b in top[i:]]   # i: includes squares

    def expand(M):
        extra = np.column_stack([M[:, a] * M[:, b] for a, b in pairs])
        return np.c_[M, extra]

    X, X_test = expand(X), expand(X_test)
    mean, std = X[tr_idx].mean(axis=0), X[tr_idx].std(axis=0)
    std[std == 0] = 1.0
    tx_tr, tx_val, tx_test = prepare(X[tr_idx]), prepare(X[val_idx]), prepare(X_test)
    print(f"added {len(pairs)} squared / interaction terms -> {tx_tr.shape[1]} features")
tx_val, y_val = prepare(X[val_idx]), y[val_idx]
tx_test = prepare(X_test)
print(f"after standardisation: {tx_tr.shape[1]} features (including the intercept)")


# ================================================================ 2. RUN MODEL
w, loss = reg_logistic_regression(y_tr, tx_tr, args.lambda_,
                                  np.zeros(tx_tr.shape[1]), args.iters, args.gamma)
if not np.all(np.isfinite(w)):
    sys.exit(f"diverged — lower --gamma (currently {args.gamma})")
print(f"trained on {len(y_tr)} rows: final loss {loss:.4f}")


# ==================================================================== 3. METRICS
def metrics(prob, truth, threshold):
    pred = (prob >= threshold).astype(float)
    tp = np.sum((pred == 1) & (truth == 1))
    fp = np.sum((pred == 1) & (truth == 0))
    fn = np.sum((pred == 0) & (truth == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, np.mean(pred == truth)


prob_tr = sigmoid(tx_tr @ w)
prob_val = sigmoid(tx_val @ w)

# The threshold is chosen on the TRAINING set, never on the validation set.
# 0.5 is the right cut-off for accuracy, not for F1 on an imbalanced problem.
grid = np.linspace(0.02, 0.60, 59)
best_thr = max(grid, key=lambda t: metrics(prob_tr, y_tr, t)[2])

print(f"\n{'':22s} {'precision':>10s} {'recall':>8s} {'F1':>8s} {'accuracy':>9s}")
for name, prob, truth, thr in [("train  @ 0.50", prob_tr, y_tr, 0.5),
                               ("val    @ 0.50", prob_val, y_val, 0.5),
                               (f"train  @ {best_thr:.2f}", prob_tr, y_tr, best_thr),
                               (f"val    @ {best_thr:.2f}", prob_val, y_val, best_thr)]:
    p, r, f, a = metrics(prob, truth, thr)
    print(f"{name:22s} {p:10.3f} {r:8.3f} {f:8.4f} {a:9.4f}")

f1_val = metrics(prob_val, y_val, best_thr)[2]
print(f"\nbaseline that always predicts 0: F1 0.0000, accuracy {1 - y.mean():.4f}")


# ================================================================ 4. SUBMISSION
# Refit on all the labelled data, with the same parameters and the threshold above.
mean, std = X.mean(axis=0), X.std(axis=0)
std[std == 0] = 1.0
tx_all, tx_test = prepare(X), prepare(X_test)
w, _ = reg_logistic_regression(y, tx_all, args.lambda_,
                               np.zeros(tx_all.shape[1]), args.iters, args.gamma)

pred = np.where(sigmoid(tx_test @ w) >= best_thr, 1, -1)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
name = f"{stamp}_{args.tag}_lam{args.lambda_:g}_g{args.gamma:g}_it{args.iters}_f1{f1_val:.4f}.csv"

with open(SUBS / name, "w") as fh:
    fh.write("Id,Prediction\n")
    for i, p in zip(ids_test, pred):
        fh.write(f"{int(i)},{int(p)}\n")

log = SUBS / "runs.csv"
if not log.exists():
    log.write_text("timestamp,tag,lambda,gamma,iters,threshold,val_f1,pct_positive,seconds,file\n")
with open(log, "a") as fh:
    fh.write(f"{stamp},{args.tag},{args.lambda_:g},{args.gamma:g},{args.iters},"
             f"{best_thr:.3f},{f1_val:.4f},{100 * (pred == 1).mean():.2f},"
             f"{time.time() - t0:.0f},{name}\n")

print(f"\nwrote submissions/{name}")
print(f"predicted positive: {int((pred == 1).sum())} ({100 * (pred == 1).mean():.2f} %)"
      f"   |   logged in submissions/runs.csv   |   {time.time() - t0:.0f}s")