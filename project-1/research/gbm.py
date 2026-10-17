"""Histogram gradient boosting for log-loss, NumPy only (level-wise trees)."""

from __future__ import annotations

import numpy as np

NB = 64  # bin 0 = missing, bins 1..63 = values


def recode_none_codes(X, ref):
    """BRFSS 'none' codes (88 / 888 / 8888 when the column max is 99 / 999 / 9999) -> 0."""
    X = X.copy()
    for j in range(X.shape[1]):
        r = ref[:, j]
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue
        mx = r.max()
        for top, none in ((99, 88), (999, 888), (9999, 8888)):
            if mx == top:
                X[X[:, j] == none, j] = 0.0
    return X


def make_bins(X, max_bins=NB - 1):
    """Quantile edges per column, fitted on X. Returns list of edge arrays."""
    edges = []
    for j in range(X.shape[1]):
        v = X[:, j]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            edges.append(np.array([]))
            continue
        u = np.unique(v)
        if len(u) <= max_bins:
            e = (u[:-1] + u[1:]) / 2
        else:
            e = np.unique(np.quantile(v, np.linspace(0, 1, max_bins + 1)[1:-1]))
        edges.append(e)
    return edges


def apply_bins(X, edges):
    B = np.zeros(X.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        v = X[:, j]
        fin = np.isfinite(v)
        B[fin, j] = (np.searchsorted(e, v[fin], side="right") + 1).astype(np.uint8)
    return B


class GBM:
    def __init__(
        self,
        n_trees=500,
        depth=5,
        lr=0.05,
        lam=1.0,
        min_hess=20.0,
        row_sub=0.5,
        col_sub=0.5,
        seed=42,
    ):
        self.n_trees, self.depth, self.lr, self.lam = n_trees, depth, lr, lam
        self.min_hess, self.row_sub, self.col_sub = min_hess, row_sub, col_sub
        self.seed = seed

    def _tree(self, B, g, h, rng):
        n, F = B.shape
        S = np.flatnonzero(rng.random(n) < self.row_sub)
        C = np.sort(rng.choice(F, max(1, int(self.col_sub * F)), replace=False))
        Fc = len(C)
        BS = B[np.ix_(S, C)].astype(np.int64)
        gS, hS = g[S], h[S]
        node_all = np.zeros(n, dtype=np.int64)
        node_S = np.zeros(len(S), dtype=np.int64)
        levels = []
        for d in range(self.depth):
            n_nodes = 1 << d
            key = (node_S[:, None] * Fc + np.arange(Fc)[None, :]) * NB + BS
            size = n_nodes * Fc * NB
            G = np.bincount(key.ravel(), weights=np.repeat(gS, Fc), minlength=size)
            H = np.bincount(key.ravel(), weights=np.repeat(hS, Fc), minlength=size)
            G = G.reshape(n_nodes, Fc, NB)
            H = H.reshape(n_nodes, Fc, NB)
            GL, HL = np.cumsum(G, axis=2), np.cumsum(H, axis=2)
            Gt, Ht = GL[:, :, -1:], HL[:, :, -1:]
            G0, H0 = G[:, :, :1], H[:, :, :1]
            lam = self.lam
            parent = Gt**2 / (Ht + lam)
            gains = []
            for GLx, HLx in ((GL, HL), (GL - G0, HL - H0)):
                GRx, HRx = Gt - GLx, Ht - HLx
                gx = GLx**2 / (HLx + lam) + GRx**2 / (HRx + lam) - parent
                gx[(HLx < self.min_hess) | (HRx < self.min_hess)] = -np.inf
                gx[:, :, -1] = -np.inf
                gains.append(gx)
            # layout per node: [missing left | missing right] x features x bins
            flat = np.stack(gains, axis=1).reshape(n_nodes, -1)
            best = np.argmax(flat, axis=1)
            ok = flat[np.arange(n_nodes), best] > 1e-12
            miss_right = ok & (best >= Fc * NB)
            rem = best % (Fc * NB)
            f_loc = rem // NB
            b_thr = np.where(ok, rem % NB, NB - 1)
            f_glob = C[f_loc]
            levels.append((f_glob, b_thr, miss_right))
            v = BS[np.arange(len(S)), f_loc[node_S]]
            node_S = 2 * node_S + (
                (v > b_thr[node_S]) | (miss_right[node_S] & (v == 0))
            )
            v = B[np.arange(n), f_glob[node_all]]
            node_all = 2 * node_all + (
                (v > b_thr[node_all]) | (miss_right[node_all] & (v == 0))
            )
        n_leaf = 1 << self.depth
        Gl = np.bincount(node_S, weights=gS, minlength=n_leaf)
        Hl = np.bincount(node_S, weights=hS, minlength=n_leaf)
        leaf = -Gl / (Hl + self.lam) * self.lr
        return levels, leaf, node_all

    @staticmethod
    def _apply(B, levels, leaf):
        n = len(B)
        node = np.zeros(n, dtype=np.int64)
        rows = np.arange(n)
        for f, b, mr in levels:
            v = B[rows, f[node]]
            node = 2 * node + ((v > b[node]) | (mr[node] & (v == 0)))
        return leaf[node]

    def fit(self, B, y, B_eval=None, checkpoints=()):
        rng = np.random.default_rng(self.seed)
        p0 = y.mean()
        self.base = float(np.log(p0 / (1 - p0)))
        F = np.full(len(y), self.base)
        Fe = None if B_eval is None else np.full(len(B_eval), self.base)
        self.trees = []
        self.snapshots = {}
        for t in range(1, self.n_trees + 1):
            p = 0.5 * (1 + np.tanh(0.5 * F))
            g, h = p - y, np.maximum(p * (1 - p), 1e-6)
            levels, leaf, node_all = self._tree(B, g, h, rng)
            F += leaf[node_all]
            self.trees.append((levels, leaf))
            if Fe is not None:
                Fe += self._apply(B_eval, levels, leaf)
                if t in checkpoints:
                    self.snapshots[t] = Fe.copy()
        return self

    def decision(self, B, n_trees=None):
        F = np.full(len(B), self.base)
        for levels, leaf in self.trees[: n_trees or len(self.trees)]:
            F += self._apply(B, levels, leaf)
        return F
