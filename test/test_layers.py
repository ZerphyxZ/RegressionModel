
import numpy as np

from numpynet.layers import Dense


def test_dense_forward():
    layer = Dense(input_size=2, output_size=1, seed=42)

    X = np.array([
        [2.0, 3.0],
        [4.0, 5.0],
    ])

    output = layer.forward(X)

    assert output.shape == (2, 1)
    assert np.all(np.isfinite(output))


def test_dense_backward():
    layer = Dense(input_size=2, output_size=1, seed=42)

    X = np.array([
        [2.0, 3.0],
        [4.0, 5.0],
    ])

    output = layer.forward(X)
    dvalues = np.ones_like(output)

    dinputs = layer.backward(dvalues)

    assert dinputs.shape == (2, 2)
    assert layer.dweights.shape == (2, 1)
    assert layer.dbiases.shape == (1, 1)


def test_dense_weight_gradient():
    layer = Dense(input_size=2, output_size=1, seed=42)

    X = np.array([[2.0, 3.0]])
    layer.forward(X)
    layer.backward(np.array([[1.0]]))

    np.testing.assert_allclose(
        layer.dweights,
        np.array([[2.0], [3.0]]),
    )


def test_dense_bias_gradient():
    layer = Dense(input_size=2, output_size=1, seed=42)

    X = np.array([[2.0, 3.0]])
    layer.forward(X)
    layer.backward(np.array([[1.0]]))

    np.testing.assert_allclose(
        layer.dbiases,
        np.array([[1.0]]),
    )
