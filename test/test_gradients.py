import numpy as np
import pytest

from numpynet.activations import ReLU
from numpynet.losses import MSE


def numerical_gradient(f, x, eps=1e-6):
    # Nudge each value a tiny bit up and down and watch how f(x) moves.
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (f(plus) - f(minus)) / (2 * eps)
    return grad


def test_mse_forward():
    loss = MSE()

    y_pred = np.array([[1.0], [2.0], [3.0]])
    y_true = np.array([[1.0], [4.0], [0.0]])

    # Errors are 0, -2, 3 -> (0 + 4 + 9) / 3
    assert np.isclose(loss.forward(y_pred, y_true), 13.0 / 3.0)


def test_mse_backward_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    y_pred = rng.normal(size=(4, 2))
    y_true = rng.normal(size=(4, 2))

    loss = MSE()
    loss.forward(y_pred, y_true)
    analytic = loss.backward()

    numeric = numerical_gradient(lambda p: MSE().forward(p, y_true), y_pred)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        MSE().forward(np.zeros((3, 1)), np.zeros(3))


def test_relu_forward():
    out = ReLU().forward(np.array([[-2.0, 0.0, 3.0]]))
    np.testing.assert_array_equal(out, [[0.0, 0.0, 3.0]])


def test_relu_backward_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3))
    upstream = rng.normal(size=(4, 3))

    relu = ReLU()
    relu.forward(x)
    analytic = relu.backward(upstream)

    # Treat "sum(relu(x) * upstream)" as the loss so its gradient is upstream * relu'(x).
    numeric = numerical_gradient(lambda v: np.sum(ReLU().forward(v) * upstream), x)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-8)
