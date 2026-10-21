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
    e = y - tx @ w
    return 1 / 2 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Gradient of ``compute_mse`` with respect to w.

    Returns:
        np.ndarray of shape (D,)
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(z):
    """Numerically stable logistic sigmoid."""
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def compute_logistic_loss(y, tx, w):
    """Negative log-likelihood of the logistic model, averaged over the N samples.

    Args:
        y: shape (N,), values in {0, 1}
        tx: shape (N, D)
        w: shape (D,)

    Returns:
        float: mean(log(1 + exp(tx @ w)) - y * (tx @ w))
    """
    z = tx @ w
    # log(1 + exp(z)) written as logaddexp(0, z) so that a large z does not overflow
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of ``compute_logistic_loss`` with respect to w.

    Returns:
        np.ndarray of shape (D,): tx.T @ (sigmoid(tx @ w) - y) / N
    """
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


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
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w -= gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


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

    w = np.array(initial_w, dtype=float)
    N = len(y)
    for _ in range(max_iters):
        i = np.random.randint(N)
        grad = compute_mse_gradient(y[i : i + 1], tx[i : i + 1], w)
        w -= gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


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
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    w = np.asarray(w).ravel()
    return w, compute_mse(y, tx, w)


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
    n, d = tx.shape
    a = tx.T @ tx + 2.0 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    w = np.asarray(w).ravel()
    return w, compute_mse(y, tx, w)


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
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


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
    w = initial_w
    for _ in range(max_iters):
        # gradient of the penalized objective: logistic gradient + 2 * lambda_ * w
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    # the returned loss does not include the penalty term
    return w, compute_logistic_loss(y, tx, w)
