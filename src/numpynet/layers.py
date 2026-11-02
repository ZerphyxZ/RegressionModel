
import numpy as np


class Dense:
    def __init__(self, input_size, output_size, seed=None):
        # Initialize the random number generator.
        rng = np.random.default_rng(seed)

        # Learnable weights and biases.
        # He init: sqrt(2/n) keeps signals from shrinking through ReLU,
        # which zeroes about half of them.
        self.weights = rng.normal(
            0.0,
            np.sqrt(2.0 / input_size),
            size=(input_size, output_size),
        )

        self.biases = np.zeros((1, output_size))

        # Values saved during the forward pass.
        self.inputs = None

        # Gradients calculated during backpropagation.
        self.dweights = None
        self.dbiases = None

    def forward(self, inputs):
        # Save inputs because backward() will need them.
        self.inputs = inputs

        # Calculate weighted sums plus biases.
        output = inputs @ self.weights + self.biases

        return output

    def backward(self, dvalues):
        # Gradient of the loss with respect to weights.
        self.dweights = self.inputs.T @ dvalues

        # Gradient of the loss with respect to biases.
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient to send to the previous layer.
        dinputs = dvalues @ self.weights.T

        return dinputs
