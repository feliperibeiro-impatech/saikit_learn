from saikit_learn.discriminant_analysis.base import _BaseDiscriminant
import numpy as np

class QDA(_BaseDiscriminant):
    def __init__(self, reg_param : float = 0.0):
        """Inicializa o QDA

        Args:
            reg_param (float): Valor somado à diagonal de cada matriz de covariância
                para evitar matrizes singulares. Deve ser maior ou igual a 0.
                Com 0.0 não aplica regularização.
        """
        super().__init__()

        if not np.isfinite(reg_param) or reg_param < 0:
            raise ValueError("reg_param deve ser finito e não negativo.")

        self.reg_param = reg_param

    def _fit_covariance(self, X : np.ndarray, Y : np.ndarray) -> None:
        """Calcula uma covariância para cada classe do QDA

        Args:
            X (np.ndarray): Matriz de atributos
            Y (np.ndarray): Vetor dos valores
        """

        self._covariance = None

        n_features = X.shape[1]
        n_classes = len(self._classes)

        covariance = np.zeros((n_classes, n_features, n_features))

        for k, c in enumerate(self._classes):
            X_c = X[Y == c]
            n_c = len(X_c)

            if n_c < 2:
                raise ValueError(f"A classe {c!r} tem {n_c} amostra(s); o QDA precisa de pelo menos 2 por classe.")

            centro = X_c - self._means[k]
            cov = centro.T @ centro / (n_c - 1) + self.reg_param * np.eye(n_features)

            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                raise ValueError(f"Covariância da classe {c!r} é singular. Use reg_param > 0 ou remova atributos colineares.")

            covariance[k] = cov

        self._covariance = covariance

    def predict(self, X : np.ndarray) -> np.ndarray:
        """Estima valores de acordo com as covariâncias calculadas

        Args:
            X (np.ndarray): Matriz de atributos

        Returns:
            np.ndarray: Vetor dos valores calculados
        """

        if self._covariance is None:
            raise RuntimeError("Chame fit antes de predict.")

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X deve ser np.ndarray, recebeu {type(X).__name__}")

        X = X.astype(np.float64, copy=False)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError(f"X deve ter shape (n_samples, {self._n_features}).")
        if not np.isfinite(X).all():
            raise ValueError("X contém valores não finitos.")

        n_samples = X.shape[0]
        n_classes = len(self._classes)

        score = np.zeros((n_samples, n_classes))

        for k in range(n_classes):
            cov = self._covariance[k]
            m = self._means[k]
            p = self._priors[k]

            _, logdet = np.linalg.slogdet(cov)

            centro = X - m 
            sol = np.linalg.solve(cov, centro.T).T
            mahalanobis = np.sum(centro * sol, axis=1)

            score[:, k] = -0.5 * logdet - 0.5 * mahalanobis + np.log(p)

        return self._classes[np.argmax(score, axis=1)]