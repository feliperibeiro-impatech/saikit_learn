from saikit_learn.utilidades.validacao import verify_X_y, NotFittedError
import numpy as np

class Lasso:
    def __init__(self, lmbd : float = 1.0, intercepto : bool = True):
        self.lmbd = lmbd
        self.intercepto = intercepto
        self._fitted = False

    def fit(self, X, y):
        pass

    def predict(self, X : np.ndarray):
        pass

