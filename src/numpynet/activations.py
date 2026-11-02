import numpy as np


class ReLU:
    def __init__(self):
        # Values saved during the forward pass.
        self.inputs = None

    def forward(self, inputs):
        # Save inputs because backward() will need them.
        self.inputs = inputs

        # Keep positive values, turn negative values into 0.
        return np.maximum(0.0, inputs)

    def backward(self, dvalues):
        # Gradient flows through where the input was positive, blocked elsewhere.
        return dvalues * (self.inputs > 0)
