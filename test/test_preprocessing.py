import numpy as np
import pytest

from numpynet.preprocessing import Scaler


def test_scaler_uses_train_numbers_on_test_data():
    X_train = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]])
    X_test = np.array([[2.0, 10.0], [8.0, 10.0]])

    scaler = Scaler().fit(X_train)

    # Train mean is 2; column 2 never changes, so it becomes 0, not nan.
    np.testing.assert_allclose(scaler.transform(X_train).mean(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(scaler.transform(X_test)[:, 1], [0.0, 0.0])

    # 8 is (8 - 2) / std(0, 2, 4) = 6 / 1.633 = 3.67 spreads above the train mean.
    assert np.isclose(scaler.transform(X_test)[1, 0], 6.0 / np.std([0.0, 2.0, 4.0]))


def test_scaler_clips_extreme_values():
    X_train = np.array([[0.0], [1.0], [2.0]])
    scaler = Scaler(clip=5.0).fit(X_train)

    np.testing.assert_allclose(scaler.transform(np.array([[1000.0], [-1000.0]])), [[5.0], [-5.0]])


def test_scaler_must_be_fit_first():
    with pytest.raises(RuntimeError):
        Scaler().transform(np.zeros((2, 2)))
