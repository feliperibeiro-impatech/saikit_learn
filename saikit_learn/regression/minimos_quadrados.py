from saikit_learn.regression.utils.validacao import verify_array, verify_X_y, NotFittedError
import numpy as np

class OLS:    
    def __init__(self, intercepto: bool = True):
        self.intercepto = intercepto
        self._fitted = False

    def _design(self, x: np.ndarray) -> np.ndarray:
        """Monta a matriz de design usada no ajuste.

        Args:
            x (np.ndarray): Matriz de atributos, de tamanho (n_samples, n_features).

        Returns:
            np.ndarray: Matriz de design, de tamanho (n_samples, n_features + 1) com intercepto ou (n_samples, n_features) sem intercepto.
        """        
        if self.intercepto:
            X = np.column_stack([np.ones(x.shape[0]), x])
            return X
        return x

    def fit(self, X: np.ndarray, y:np.ndarray):
        """Treina o modelo dos X de acordo com os valores y dados.

        Args:
            X (np.ndarray): Atributos de treino, de shape (n_samples, n_features).
            y (np.ndarray): Valores-alvo, de shape (n_samples,).
        """           
        X, y =  verify_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        X = self._design(X)
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            raise ValueError("X tem colunas linearmente dependentes; OLS não tem solução única.")
        
        self.intercept_ = float(beta[0]) if self.intercepto else 0.0
        self.coef_ = beta[1:] if self.intercepto else beta
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Estima os valores de X 

        Args:
            X (np.ndarray): Valores a serem estimados, de tamanho (n_samples, n_features).

        Returns:
            np.ndarray: Valores estimados pelo método, de tamanho (n_samples,).
        """  
        if not self._fitted:
            raise NotFittedError(
                f'{self.__class__.__name__} não foi treinado ainda, chame fit.'
                )
        
        X = verify_array(X)

        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f'a dimensão das features difere; necessário: {self.n_features_in_}; obtido: {X.shape[1]}'
                             )
        return X @ self.coef_ + self.intercept_