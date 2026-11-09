from saikit_learn.regression.utils.validacao import verify_array, verify_X_y, NotFittedError
from saikit_learn.regression.minimos_quadrados import OLS
import numpy as np

class PolReg:
    def __init__(self, d : int, intercepto: bool = True):
        """Inicializa o modelo de regressão polinomial.

        Args:
            d (int): Grau do polinômio ajustado.
            intercepto (bool, opcional): Se True, inclui o termo constante (intercepto) no modelo. Padrão é True.
        """        
        self.d = d
        self.intercepto = intercepto
        self._fitted = False

    def _design(self, X : np.ndarray) -> np.ndarray:
        """Monta a matriz de potências usada no ajuste.

        Args:
            X (np.ndarray): Atributo de entrada, de tamanho (n_samples, 1).

        Returns:
            np.ndarray: Matriz de potências, de tamanho (n_samples, d).
        """        
        return np.hstack([X ** k for k in range(1, self.d + 1)])

    def fit(self, X : np.ndarray, y : np.ndarray):
        """Treina o modelo polinomial de grau d com os dados fornecidos.

        Args:
            X (np.ndarray): Atributo de treino, de tamanho (n_samples, 1).
            y (np.ndarray): Valores-alvo, de tamanho (n_samples,).
        """        
        if not isinstance(self.d, (int, np.integer)) or isinstance(self.d, bool) or self.d < 1:
            raise ValueError(
                f'd deve ser um número inteiro positivo; obtido: {self.d}'
            )
        
        X, y = verify_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if X.shape[1] != 1:
            raise ValueError(
                f'a dimensão das features está incorreta; necessário: 1; obtido: {self.n_features_in_}'
            )

        self.model_ = OLS(intercepto=self.intercepto).fit(self._design(X), y)
        self._fitted = True
        return self

    def predict(self, X : np.ndarray) -> np.ndarray:
        """Estima os valores de saída para os valores de X.

        Args:
            X (np.ndarray): Valores a serem estimados, de shape (n_samples, 1).

        Returns:
            np.ndarray: Valores estimados pelo polinômio ajustado, de tamanho (n_samples,).
        """        
        if not self._fitted:
            raise NotFittedError(
                f'{self.__class__.__name__} não foi fittado ainda'
                )
        X = verify_array(X)
        if self.n_features_in_ != X.shape[1]:
            ValueError(
                f'a dimensão das features distingue; necessário: {self.n_features_in_}; obtido: {X.shape[1]}'
            )
        return self.model_.predict(self._design(X))