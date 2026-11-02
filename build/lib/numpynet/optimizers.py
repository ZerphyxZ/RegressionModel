import numpy as np


class SGD:
    def __init__(self, learning_rate=0.1):
        # How big a step to take each update.
        self.learning_rate = learning_rate

    def update(self, layer):
        # Step each weight a little against its gradient (downhill).
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1  # How much of the past direction to keep (momentum).
        self.beta2 = beta2  # How much of the past gradient size to keep.
        self.eps = eps      # Tiny number so we never divide by zero.

        # Memory for each layer: step count, running direction, running size.
        self.state = {}

    def update(self, layer):
        if layer not in self.state:
            self.state[layer] = {
                "t": 0,
                "m": [np.zeros_like(layer.weights), np.zeros_like(layer.biases)],
                "v": [np.zeros_like(layer.weights), np.zeros_like(layer.biases)],
            }
        s = self.state[layer]
        s["t"] += 1

        params = [(layer.weights, layer.dweights), (layer.biases, layer.dbiases)]
        for i, (param, grad) in enumerate(params):
            # Running average of the gradient: which way we've been heading.
            s["m"][i] = self.beta1 * s["m"][i] + (1 - self.beta1) * grad
            # Running average of the squared gradient: how big the gradients are.
            s["v"][i] = self.beta2 * s["v"][i] + (1 - self.beta2) * grad**2

            # Both averages start at 0, so they're too small early on. Fix that.
            m_hat = s["m"][i] / (1 - self.beta1 ** s["t"])
            v_hat = s["v"][i] / (1 - self.beta2 ** s["t"])

            # Step in the average direction, scaled down where gradients are big.
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
