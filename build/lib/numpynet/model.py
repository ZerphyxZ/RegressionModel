import numpy as np


class Model:
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def predict(self, X):
        # Forward: pass the data through every layer in order.
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def evaluate(self, X, y):
        # Measure the loss without changing any weights.
        return self.loss.forward(self.predict(X), y)

    def train_epoch(self, X, y, batch_size, rng):
        # Shuffle so every epoch sees the batches in a new random order.
        order = rng.permutation(len(X))
        total_loss = 0.0

        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]

            loss = self.loss.forward(self.predict(X[batch]), y[batch])

            # Backward: same path in reverse, passing the gradient along.
            grad = self.loss.backward()
            for layer in reversed(self.layers):
                grad = layer.backward(grad)

            # Update: only layers with weights (Dense) get changed.
            for layer in self.layers:
                if hasattr(layer, "weights"):
                    self.optimizer.update(layer)

            total_loss += loss * len(batch)

        # Average loss over the whole epoch.
        return total_loss / len(X)
