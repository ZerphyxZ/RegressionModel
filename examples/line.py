import numpy as np

from numpynet.layers import Dense
from numpynet.losses import MSE
from numpynet.optimizers import SGD


# Fake data: y = 3x + 2 plus a little noise.
rng = np.random.default_rng(0)
X = rng.uniform(-1.0, 1.0, size=(200, 1))
y = 3.0 * X + 2.0 + rng.normal(0.0, 0.1, size=(200, 1))

layer = Dense(input_size=1, output_size=1, seed=42)
loss_fn = MSE()
optimizer = SGD(learning_rate=0.1)

for epoch in range(201):
    # Forward: make predictions and measure how wrong they are.
    y_pred = layer.forward(X)
    loss = loss_fn.forward(y_pred, y)

    # Backward: work out which way each weight should move.
    dvalues = loss_fn.backward()
    layer.backward(dvalues)

    # Update: let the optimizer nudge the weights downhill.
    optimizer.update(layer)

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss:.4f}")

print(f"learned: y = {layer.weights[0, 0]:.3f}x + {layer.biases[0, 0]:.3f}  (target: 3x + 2)")
