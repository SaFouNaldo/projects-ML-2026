"""Interactions pairwise sur les top-K colonnes (|w| IRLS).

Fit sur train uniquement (indices + scale des produits), puis transform test.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-np.clip(z[pos], -50, 50)))
    ez = np.exp(np.clip(z[~pos], -50, 50))
    out[~pos] = ez / (1.0 + ez)
    return out


def _logreg_irls(
    X: np.ndarray,
    y01: np.ndarray,
    lam: float = 1.0,
    n_iter: int = 25,
    tol: float = 1e-8,
    bias_idx: int = 0,
) -> np.ndarray:
    d = X.shape[1]
    w = np.zeros(d, dtype=np.float64)
    R = lam * np.eye(d)
    R[bias_idx, bias_idx] = 0.0
    for _ in range(n_iter):
        p = _sigmoid(X @ w)
        g = X.T @ (p - y01) + R @ w
        W = p * (1.0 - p)
        H = (X * W[:, None]).T @ X + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    return w


class TopKPairInteractions:
    """Ajoute les produits x_i * x_j pour les K colonnes au plus fort |w|."""

    def __init__(
        self, k: int = 12, lam: float = 1.0, bias_idx: int = 0, scale: bool = True
    ):
        self.k = k
        self.lam = lam
        self.bias_idx = bias_idx
        self.scale = scale
        self.top_idx_: np.ndarray | None = None
        self.pairs_: list[tuple[int, int]] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.names_: list[str] = []

    def fit(self, X: np.ndarray, y01: np.ndarray) -> "TopKPairInteractions":
        w = _logreg_irls(X, y01, lam=self.lam, bias_idx=self.bias_idx)
        score = np.abs(w.copy())
        score[self.bias_idx] = -1.0
        k = min(self.k, X.shape[1] - 1)
        self.top_idx_ = np.argsort(-score)[:k]
        self.pairs_ = []
        self.names_ = []
        for a in range(k):
            for b in range(a + 1, k):
                i, j = int(self.top_idx_[a]), int(self.top_idx_[b])
                self.pairs_.append((i, j))
                self.names_.append(f"inter_{i}x{j}")
        Z = self._raw_products(X)
        if self.scale and Z.shape[1] > 0:
            self.mean_ = Z.mean(axis=0)
            self.std_ = Z.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
        else:
            self.mean_ = np.zeros(Z.shape[1])
            self.std_ = np.ones(Z.shape[1])
        return self

    def _raw_products(self, X: np.ndarray) -> np.ndarray:
        if not self.pairs_:
            return np.zeros((len(X), 0), dtype=np.float64)
        cols = [X[:, i] * X[:, j] for i, j in self.pairs_]
        return np.column_stack(cols)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.top_idx_ is not None and self.mean_ is not None
        Z = self._raw_products(X)
        if self.scale and Z.shape[1] > 0:
            Z = (Z - self.mean_) / self.std_
        return np.hstack([X, Z])

    def fit_transform(self, X: np.ndarray, y01: np.ndarray) -> np.ndarray:
        self.fit(X, y01)
        return self.transform(X)
