import numpy as np

from numpynet.layers import Dense
from numpynet.optimizers import Adam


def test_adam_first_step_moves_each_weight_by_learning_rate():
    layer = Dense(input_size=2, output_size=2, seed=0)
    before = layer.weights.copy()

    # Gradients of very different sizes and signs.
    layer.dweights = np.array([[100.0, -0.01], [3.0, -50.0]])
    layer.dbiases = np.array([[0.5, -2.0]])

    Adam(learning_rate=0.1).update(layer)

    # On the first step Adam ignores gradient size: every weight moves by
    # exactly the learning rate, against the sign of its gradient.
    np.testing.assert_allclose(before - layer.weights, 0.1 * np.sign(layer.dweights), rtol=1e-5)
    np.testing.assert_allclose(layer.biases, [[-0.1, 0.1]], rtol=1e-5)
