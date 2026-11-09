from saikit_learn.discriminant_analysis.base import _BaseDiscriminant
import numpy as np

class LDA(_BaseDiscriminant):
    def _fit_covariance(self, X : np.ndarray, Y : np.ndarray) -> None:
        """Calcula covariância do LDA

        Args:
            X (np.ndarray): Matriz de atributos
            Y (np.ndarray): Vetor dos valores
        """        

        n_samples, n_features = X.shape
        n_classes = len(self._classes)

        scatter = np.zeros((n_features, n_features))
        for k, c in enumerate(self._classes):
            centro = X[Y == c] - self._means[k]
            scatter += centro.T @ centro

        self._covariance = scatter / (n_samples - n_classes)

    def predict(self, X : np.ndarray) -> np.ndarray:
        """Estima valores de acordo com a covariância calculada

        Args:
            X (np.ndarray): Matriz de atributos

        Returns:
            np.ndarray: Vetor dos valores calculados
        """   
             
        if self._covariance is None:
            raise RuntimeError("Chame fit antes de predict.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError(f"X deve ter shape (n_samples, {self._n_features}).")
        
        cov_inv = np.linalg.inv(self._covariance)
        n_classes = len(self._classes)

        predictions = []

        for x in X:
            score = np.zeros(n_classes)

            for k in range(n_classes):
                m = self._means[k]
                p = self._priors[k]
                score[k] = x @ cov_inv @ m - 0.5 * m @ cov_inv @ m + np.log(p)

            predictions.append(self._classes[np.argmax(score)])
 
        return np.array(predictions)

