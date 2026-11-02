import numpy as np


class Scaler:
    def __init__(self, clip=None):
        # Optional cap for extreme values after scaling, e.g. clip=5 -> [-5, 5].
        self.clip = clip

        # Learned from the training data in fit().
        self.mean = None
        self.std = None

    def fit(self, X):
        # Learn the centre and spread of each column. Call on train data only.
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # A column that never changes has spread 0; avoid dividing by 0.
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("call fit() on the training data first")

        # Centre on 0 with a spread of 1, using the numbers learned in fit().
        X = (X - self.mean) / self.std

        if self.clip is not None:
            X = np.clip(X, -self.clip, self.clip)
        return X
