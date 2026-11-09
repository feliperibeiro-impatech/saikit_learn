from utils.validacao import verify_X_y, verify_array, NotFittedError
import numpy as np

class Lasso:
    def __init__(self, lmbd : float = 1.0, intercepto : bool = True):
        """Regressão Lasso por descida por coordenadas.

        Args:
            lmbd (float, opcional): Força da penalidade L1 (>= 0).. Padrão é 1.0.
            intercepto (bool, opcional): Se True, ajusta um intercepto não penalizado. Padrão é True.
        """
        if (isinstance(lmbd, bool) 
                or not isinstance(lmbd, (int, float, np.integer, np.floating))
                or not np.isfinite(lmbd) or lmbd < 0):
            raise ValueError("lmbd deve ser um número real finito e não negativo.")
        
        self.lmbd = lmbd
        self.intercepto = intercepto
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina o modelo dos X de acordo com os valores y dados.

        Args:
            X (np.ndarray): Atributos com tamanho (n_samples, n_features)
            y (np.ndarray): Valores alvos de tamanho (n_sample)
        """        
        X, y = verify_X_y(X, y)
        n, d = X.shape
        self.n_features_in_ = d

        if self.intercepto:
            x_mean, y_mean = X.mean(axis=0), y.mean()
        else:
            x_mean, y_mean = np.zeros(d), 0.0
        Xc, yc = X - x_mean, y - y_mean

        col_sq = (Xc ** 2).sum(axis=0) / n
        w = np.zeros(d)
        residual = yc.copy()

        for _ in range(1000):
            w_original = w.copy()
            for j in range(d):
                if col_sq[j] == 0:
                    continue
                w_old = w[j]
                rho = Xc[:, j] @ residual / n + col_sq[j] * w_old
                w_new = np.sign(rho) * max(abs(rho) - self.lmbd, 0.0) / col_sq[j]
                if w_new != w_old:
                    residual -= Xc[:, j] * (w_new - w_old)
                    w[j] = w_new
            if np.allclose(w, w_original, atol=1e-8, rtol=0):
                break

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        self._fitted = True

        return self
    
    def predict(self, X : np.ndarray):
        """Estima os alvos para X de tamanho (n_samples, n_features).

        Args:
            X (np.ndarray): Atributos, shape (n_samples, n_features).

        Returns:
            np.ndarray: Valores estimados, shape (n_samples,).
        """
        if not self._fitted:
            raise NotFittedError(f"{self.__class__.__name__} não foi ajustado ainda.")

        X = verify_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Número de features difere: esperado {self.n_features_in_}, obtido {X.shape[1]}."
            )

        return X @ self.coef_ + self.intercept_

