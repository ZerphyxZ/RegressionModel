import numpy as np


class MSE:
    def __init__(self):
        # Values saved during the forward pass.
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            # (n, 1) vs (n,) would silently broadcast to (n, n).
            raise ValueError(f"shape mismatch: {y_pred.shape} vs {y_true.shape}")

        # Save both because backward() will need them.
        self.y_pred = y_pred
        self.y_true = y_true

        # Average squared error over every sample and output.
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        # d/dy_pred of mean((y_pred - y_true)^2) = 2 * (y_pred - y_true) / N
        return 2 * (self.y_pred - self.y_true) / self.y_pred.size
