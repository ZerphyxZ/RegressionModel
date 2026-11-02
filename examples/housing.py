import numpy as np
from sklearn.datasets import fetch_california_housing

from numpynet import MSE, Adam, Dense, Model, ReLU, Scaler


# Real data: 20,640 California districts, 8 inputs each -> median house price.
# Prices are in units of $100,000 (so 2.5 means $250,000).
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)  # (n,) -> (n, 1) so it matches the predictions.

# Hold back 20% as a test set the model never trains on.
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
split = int(0.8 * len(X))
train, test = order[:split], order[split:]
X_train, y_train = X[train], y[train]
X_test, y_test = X[test], y[test]

# Scale inputs using train numbers only, then cap extreme values at +/-5.
# A few districts are wild (e.g. 1,243 people per house) and their huge
# values would blow up the gradients.
scaler = Scaler(clip=5.0).fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Network: 8 inputs -> 64 -> ReLU -> 64 -> ReLU -> 1 price.
model = Model(
    layers=[
        Dense(input_size=8, output_size=64, seed=1),
        ReLU(),
        Dense(input_size=64, output_size=64, seed=2),
        ReLU(),
        Dense(input_size=64, output_size=1, seed=3),
    ],
    loss=MSE(),
    optimizer=Adam(learning_rate=0.001),
)

for epoch in range(1, 61):
    train_loss = model.train_epoch(X_train, y_train, batch_size=32, rng=rng)

    if epoch == 1 or epoch % 10 == 0:
        test_loss = model.evaluate(X_test, y_test)
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  test loss {test_loss:.4f}")


def typical_error_dollars(mse):
    # sqrt turns squared error back into price units, then * $100,000.
    return np.sqrt(mse) * 100_000


# Baseline: ignore the inputs and always guess the average training price.
baseline_mse = np.mean((y_test - y_train.mean()) ** 2)
model_mse = model.evaluate(X_test, y_test)

print(f"\nalways guess the average: typically off by ${typical_error_dollars(baseline_mse):,.0f}")
print(f"our network:                typically off by ${typical_error_dollars(model_mse):,.0f}")

# Half of all districts are predicted closer than this.
errors = np.abs(model.predict(X_test) - y_test) * 100_000
print(f"median error:             ${np.median(errors):,.0f}")

# A few real test districts side by side.
print("\n  predicted     actual")
for pred, actual in zip(model.predict(X_test[:5])[:, 0], y_test[:5, 0]):
    print(f"  ${pred * 100_000:>9,.0f}  ${actual * 100_000:>9,.0f}")
