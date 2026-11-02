import numpy as np

from numpynet.activations import ReLU
from numpynet.layers import Dense
from numpynet.losses import MSE
from numpynet.model import Model
from numpynet.optimizers import SGD


def test_model_learns_a_line():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(100, 1))
    y = 3.0 * X + 2.0

    model = Model(
        layers=[Dense(1, 8, seed=1), ReLU(), Dense(8, 1, seed=2)],
        loss=MSE(),
        optimizer=SGD(learning_rate=0.05),
    )

    start_loss = model.evaluate(X, y)
    for _ in range(200):
        model.train_epoch(X, y, batch_size=16, rng=rng)

    assert model.evaluate(X, y) < start_loss / 100
