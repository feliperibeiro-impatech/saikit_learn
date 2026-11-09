import numpy as np

class StandardScaler:
    """Padroniza cada coluna para média zero e desvio padrão um."""

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        """Inicializa o scaler escolhendo quais estatísticas serão aplicadas.

        Args:
            with_mean: Se True, centraliza os dados subtraindo a média.
            with_std: Se True, escala os dados dividindo pelo desvio padrão.
        """
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> StandardScaler:
        """Estima a média e o desvio padrão de cada coluna a partir de X.

        Args:
            X: Matriz de dados com shape (n_amostras, n_features).
            y: Ignorado, existe apenas para compatibilidade com o Pipeline.

        Returns:
            A própria instância ajustada.
        """
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        std = X.std(axis=0) if self.with_std else np.ones(X.shape[1])
        self.scale_ = np.where(std == 0, 1.0, std)  # evita divisão por zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica a padronização usando as estatísticas estimadas no fit.

        Args:
            X: Matriz de dados com shape (n_amostras, n_features).

        Returns:
            Matriz padronizada com o mesmo shape de X.
        """
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Ajusta o scaler em X e retorna X padronizado.

        Args:
            X: Matriz de dados com shape (n_amostras, n_features).
            y: Ignorado, existe apenas para compatibilidade com o Pipeline.

        Returns:
            Matriz padronizada com o mesmo shape de X.
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Desfaz a padronização, devolvendo os dados à escala original.

        Args:
            Z: Matriz padronizada com shape (n_amostras, n_features).

        Returns:
            Matriz na escala original com o mesmo shape de Z.
        """
        return np.asarray(Z, dtype=float) * self.scale_ + self.mean_


class Pipeline:
    """Encadeia transformadores e um estimador final em um único objeto."""

    def __init__(self, steps: list[tuple[str, object]]) -> None:
        """Inicializa o pipeline com a sequência de etapas.

        Args:
            steps: Lista de pares (nome, objeto), em que todos os objetos exceto
                o último implementam fit_transform/transform e o último
                implementa fit/predict.
        """
        self.steps = steps

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Pipeline:
        """Ajusta cada transformador em sequência e depois o estimador final.

        Args:
            X: Matriz de dados com shape (n_amostras, n_features).
            y: Vetor alvo com shape (n_amostras,), repassado às etapas.

        Returns:
            A própria instância ajustada.
        """
        for _, step in self.steps[:-1]:
            X = step.fit_transform(X, y)
        self.steps[-1][1].fit(X, y)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica todos os transformadores já ajustados, sem o estimador final.

        Args:
            X: Matriz de dados com shape (n_amostras, n_features).

        Returns:
            Matriz transformada pelas etapas intermediárias.
        """
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transforma X pelas etapas intermediárias e prediz com o estimador final.

        Args:
            X: Matriz de dados com shape (n_amostras, n_features).

        Returns:
            Vetor de predições com shape (n_amostras,).
        """
        return self.steps[-1][1].predict(self._transform(X))