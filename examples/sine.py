import numpy as np

from numpynet import MSE, SGD, Dense, Model, ReLU, Scaler


# Fake data: a curved line, y = sin(x), plus a little noise.
rng = np.random.default_rng(0)
X = rng.uniform(-np.pi, np.pi, size=(1000, 1))
y = np.sin(X) + rng.normal(0.0, 0.1, size=(1000, 1))

# Hold back 20% as a test set the model never trains on.
order = rng.permutation(len(X))
split = int(0.8 * len(X))
train, test = order[:split], order[split:]
X_train, y_train = X[train], y[train]
X_test, y_test = X[test], y[test]

# Scale inputs to mean 0, spread 1. Use train numbers only, then apply the
# same numbers to the test set, so no test info leaks into training.
scaler = Scaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Network: 1 input -> 32 hidden units -> ReLU -> 1 output.
model = Model(
    layers=[
        Dense(input_size=1, output_size=32, seed=1),
        ReLU(),
        Dense(input_size=32, output_size=1, seed=2),
    ],
    loss=MSE(),
    optimizer=SGD(learning_rate=0.05),
)

for epoch in range(301):
    train_loss = model.train_epoch(X_train, y_train, batch_size=32, rng=rng)

    if epoch % 30 == 0:
        test_loss = model.evaluate(X_test, y_test)
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  test loss {test_loss:.4f}")

# Compare a few predictions with the real curve.
X_check = np.array([[-3.0], [-1.5], [0.0], [1.5], [3.0]])
y_check = model.predict(scaler.transform(X_check))
for x_val, pred in zip(X_check[:, 0], y_check[:, 0]):
    print(f"x = {x_val:5.1f}   predicted {pred:6.3f}   sin(x) = {np.sin(x_val):6.3f}")
