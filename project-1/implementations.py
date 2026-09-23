"""The six machine learning methods required by Project 1.

Only NumPy and the Python standard library are allowed here.
Every function returns ``(w, loss)``:
    w    : np.ndarray, shape (D,) -- the LAST weight vector of the method
    loss : float                  -- the corresponding loss, WITHOUT the
                                     regularization term for the penalized methods

Conventions used across the whole project
-----------------------------------------
y   : labels, shape (N,). {0, 1} for the logistic methods, real-valued otherwise.
tx  : features, shape (N, D), bias column already included.
MSE : 0.5 * mean((y - tx @ w) ** 2)          (the 0.5 matches the lecture notes)
Log : mean(log(1 + exp(tx @ w)) - y * (tx @ w))
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers  (MSE pair: Rania -- logistic pair: Yahya)
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Mean squared error with the 0.5 factor.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        w: shape (D,)

    Returns:
        float: 0.5 * mean((y - tx @ w) ** 2)
    """
    raise NotImplementedError


def compute_mse_gradient(y, tx, w):
    """Gradient of ``compute_mse`` with respect to w.

    Returns:
        np.ndarray of shape (D,)
    """
    raise NotImplementedError


def sigmoid(t):
    """Numerically stable logistic sigmoid, applied element-wise."""
    raise NotImplementedError


def compute_logistic_loss(y, tx, w):
    """Negative log-likelihood of the logistic model, averaged over the N samples.

    Args:
        y: shape (N,), values in {0, 1}
    """
    raise NotImplementedError


def compute_logistic_gradient(y, tx, w):
    """Gradient of ``compute_logistic_loss`` with respect to w."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Linear regression -- gradient descent            (owner: Rania)
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        initial_w: shape (D,), starting point
        max_iters: int, number of steps (0 means: return initial_w and its loss)
        gamma: float, step size

    Returns:
        (w, loss): last weight vector of shape (D,) and its MSE (a scalar).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. Linear regression -- stochastic gradient descent  (owner: Rania)
# ---------------------------------------------------------------------------


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent with batch size 1.

    At each step, sample ONE datapoint and take a gradient step on it.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        initial_w: shape (D,)
        max_iters: int
        gamma: float

    Returns:
        (w, loss): last weight vector and its MSE computed on the FULL dataset.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Least squares -- normal equations                 (owner: Sara)
# ---------------------------------------------------------------------------


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Solve ``(tx.T @ tx) w = tx.T @ y``. Anything from ``numpy.linalg`` is allowed
    except ``numpy.linalg.lstsq``.

    Returns:
        (w, loss): solution of shape (D,) and its MSE.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. Ridge regression -- normal equations              (owner: Sara)
# ---------------------------------------------------------------------------


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Solve ``(tx.T @ tx + 2 * N * lambda_ * I) w = tx.T @ y``.

    Args:
        lambda_: float, regularization parameter

    Returns:
        (w, loss): solution of shape (D,) and its MSE WITHOUT the penalty term.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 5. Logistic regression -- gradient descent           (owner: Yahya)
# ---------------------------------------------------------------------------


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: shape (N,), values in {0, 1}
        tx: shape (N, D)
        initial_w: shape (D,)
        max_iters: int
        gamma: float

    Returns:
        (w, loss): last weight vector and its logistic loss.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 6. Regularized logistic regression -- gradient descent  (owner: Yahya)
# ---------------------------------------------------------------------------


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    The objective is ``logistic_loss(y, tx, w) + lambda_ * ||w||^2`` but the
    RETURNED loss must not include the penalty term.

    Args:
        y: shape (N,), values in {0, 1}
        lambda_: float, regularization parameter

    Returns:
        (w, loss): last weight vector and its logistic loss without the penalty.
    """
    raise NotImplementedError
