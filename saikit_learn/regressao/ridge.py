from saikit_learn.utilidades.validacao import verify_array, verify_X_y, NotFittedError
import numpy as np

class Ridge:
    def __init__(self, lmbd: float = 1.0, intercepto : bool = True):
        """Inicializa o modelo de regressão Ridge.

        Args:
            lmbd (float, opcional): Força da penalização L2 aplicada aos coeficientes. Valores maiores encolhem mais e 0 equivale ao OLS. Padrão é 1.0.
            intercepto (bool, opcional): Se True, inclui o termo constante (intercepto) no modelo, sem penalizá-lo. Padrão é True.
        """        
        self.lmbd = lmbd
        self.intercepto = intercepto
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina o modelo Ridge com os dados fornecidos.

        Args:
            X (np.ndarray): Atributos de treino, de shape (n_samples, n_features).
            y (np.ndarray): Valores-alvo, de shape (n_samples,).
        """        
        if isinstance(self.lmbd, bool):
            raise ValueError(
                f'lambda deve ser um número; obtido: {type(self.lmbd)}'
            )

        try:
            lmbd = float(self.lmbd)

        except (TypeError, ValueError):
            raise ValueError(
                f'lambda deve ser um número; obtido: {type(self.lmbd)}'
            )

        if not np.isfinite(lmbd) or isinstance(lmbd, bool) or lmbd < 0:
            raise ValueError(
                f'lambda deve ser um número real não negativo; obtido: {self.lmbd}'
            )

        X, y = verify_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.intercepto:
            # não podemos penalizar o intercepto, então devemos cetralizar,
            # treinar, e depois encontrar o intercepto
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            Xprime = X - X_mean
            yprime = y - y_mean

        else:
            Xprime, yprime = X, y

        self.coef_ = np.linalg.solve(Xprime.T @ Xprime + self.lmbd * np.eye(self.n_features_in_), Xprime.T @ yprime)
        self.intercept_ = float(y_mean - X_mean @ self.coef_) if self.intercepto else 0.0
        self._fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Estima os valores de saída para os valores de X.

        Args:
            X (np.ndarray): Valores a serem estimados, de tamanho n_samples, n_features).

        Returns:
            np.ndarray: Valores estimados pelo modelo, de tamanho (n_samples,).
        """        
        X = verify_array(X)

        if not self._fitted:
            raise NotFittedError(f'{self.__class__.__name__} não foi fittada ainda')

        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f'a dimensão das features está incorreta; necessário: {self.n_features_in_}; obtido: {X.shape[1]}'
            )

        return X @ self.coef_ + self.intercept_