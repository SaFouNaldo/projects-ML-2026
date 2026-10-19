"""Generic encoding of all raw columns + full-batch Adam logistic regression (L2)."""

from __future__ import annotations

import numpy as np

from implementations import sigmoid

DROP = {"SEQNO", "_PSU", "IDATE", "IYEAR", "IDAY", "_STSTR", "CTELENUM", "CTELNUM1"}
FORCE_CATEGORICAL = {"_STATE"}
MIN_LEVEL = 300
MAX_CAT_UNIQUE = 30
N_QUANT = 64


class WideEncoder:
    """Generic encoding of all raw columns (except DROP), fitted on the training rows.

    For each column: a missing indicator if the column is partly empty; one-hot of
    the values seen at least MIN_LEVEL times (small categorical columns), or of the
    frequent "spike" values such as the codes 77 / 88 / 99; for the remaining
    numeric values, their quantile rank u in [0, 1] and (u - 0.5)^2. Then
    standardisation.
    """

    def fit(self, X, hdr):
        """Chooses the encoding of each column from the training matrix X (columns hdr)
        and learns the standardisation.
        """
        self.specs = []
        n = len(X)
        for j, name in enumerate(hdr):
            if name in DROP:
                continue
            v = X[:, j]
            fin = np.isfinite(v)
            miss = 1.0 - fin.mean()
            if fin.sum() < MIN_LEVEL:
                continue
            spec = {
                "j": j,
                "name": name,
                "isna": MIN_LEVEL / n < miss < 1 - MIN_LEVEL / n,
            }
            vals, counts = np.unique(v[fin], return_counts=True)
            if len(vals) <= MAX_CAT_UNIQUE or name in FORCE_CATEGORICAL:
                spec["levels"] = vals[counts >= MIN_LEVEL]
                spec["numeric"] = False
            else:
                spike = counts >= 0.01 * n
                spec["levels"] = vals[spike]
                rest = v[fin & ~np.isin(v, vals[spike])]
                if len(rest) < MIN_LEVEL:
                    spec["numeric"] = False
                else:
                    spec["numeric"] = True
                    spec["q"] = np.unique(
                        np.quantile(rest, np.linspace(0, 1, N_QUANT + 1))
                    )
                    spec["spikes"] = vals[spike]
            self.specs.append(spec)
        Z = self._raw(X)
        self.mean = Z.mean(axis=0)
        self.std = Z.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def _raw(self, X):
        """Encoded matrix (float32) before standardisation; also sets self.names."""
        cols, names = [], []
        for s in self.specs:
            v = X[:, s["j"]]
            fin = np.isfinite(v)
            if s["isna"]:
                cols.append((~fin).astype(np.float32))
                names.append(f"{s['name']}__na")
            for lv in s["levels"]:
                cols.append((v == lv).astype(np.float32))
                names.append(f"{s['name']}={lv:g}")
            if s["numeric"]:
                ok = fin & ~np.isin(v, s["spikes"])
                q = s["q"]
                u = np.interp(
                    np.where(ok, v, q[len(q) // 2]), q, np.linspace(0, 1, len(q))
                )
                u = np.where(ok, u, 0.5).astype(np.float32)
                cols.append(u)
                cols.append(((u - 0.5) ** 2).astype(np.float32))
                names += [f"{s['name']}__u", f"{s['name']}__u2"]
        self.names = names
        return np.column_stack(cols).astype(np.float32)

    def transform(self, X):
        """Encoded and standardised matrix of X."""
        Z = self._raw(X)
        Z -= self.mean.astype(np.float32)
        Z /= self.std.astype(np.float32)
        return Z


def adam_logistic(Z, y, lam=1e-4, iters=400, lr=0.05, w0=None, b0=None, verbose=False):
    """Mean log-loss + lam * ||w||^2, full batch Adam. Returns (w, b)."""
    n, d = Z.shape
    w = np.zeros(d, dtype=np.float32) if w0 is None else w0.astype(np.float32).copy()
    b = float(np.log(y.mean() / (1 - y.mean()))) if b0 is None else float(b0)
    m = np.zeros(d, dtype=np.float32)
    v = np.zeros(d, dtype=np.float32)
    mb = vb = 0.0
    y32 = y.astype(np.float32)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        z = Z @ w + b
        p = sigmoid(z).astype(np.float32)  # stay in float32: Z is large
        r = (p - y32) / n
        g = Z.T @ r + 2 * lam * w
        gb = float(r.sum())
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mb = b1 * mb + (1 - b1) * gb
        vb = b2 * vb + (1 - b2) * gb * gb
        decay = 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * t / iters))
        lr_t = lr * decay * np.sqrt(1 - b2**t) / (1 - b1**t)
        w -= lr_t * m / (np.sqrt(v) + eps)
        b -= lr_t * mb / (np.sqrt(vb) + eps)
        if verbose and t % 100 == 0:
            loss = np.mean(np.logaddexp(0, z) - y32 * z)
            print(f"    iter {t} loss {loss:.5f}", flush=True)
    return w, b
