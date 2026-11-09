import numpy as np


class LogisticRegression:
    """Regressão Logística binária treinada por Gradient Descent.

    Minimiza a Binary Cross-Entropy média, com penalidade L2 opcional nos
    coeficientes: J(w, b) = BCE(w, b) + (alpha / 2) * ||w||^2


    Attributes:
        classes_ (np.ndarray): As duas classes, em ordem crescente.
            classes_[1] é a classe positiva.
        coef_ (np.ndarray): Coeficientes, com shape (n_features,).
        intercept_ (float): Intercepto do modelo.
        n_features_in_ (int): Número de atributos vistos no fit.
        n_iter_ (int): Número de iterações realizadas no fit.
        converged_ (bool): True se o critério de parada pelo gradiente foi
            atingido antes de max_iter.
        loss_history (list[float]): Valor de J em cada iteração, calculado
            com os parâmetros anteriores à atualização daquela iteração.
    """

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-6, alpha: float = 0.0):
        """Inicializa a Regressão Logística.

        Args:
            learning_rate (float): Taxa de aprendizado do Gradient Descent.
            max_iter (int): Número máximo de iterações.
            tol (float): Tolerância para o critério de parada. O treino
                termina quando a norma do gradiente fica abaixo de tol.
            alpha (float): Força da regularização L2. Com 0.0 (padrão) não
                há regularização. Quanto maior o alpha, menor deve ser a
                learning_rate para o treino não divergir.
        """

        # bool é subclasse de int em Python, por isso é recusado à parte.
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float, np.integer, np.floating)):
            raise ValueError("learning_rate deve ser um número.")

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate deve ser finito e positivo.")

        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
            raise ValueError("max_iter deve ser um inteiro.")

        if max_iter <= 0:
            raise ValueError("max_iter deve ser um inteiro positivo.")

        if isinstance(tol, bool) or not isinstance(tol, (int, float, np.integer, np.floating)):
            raise ValueError("tol deve ser um número.")

        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol deve ser finito e não negativo.")

        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise ValueError("alpha deve ser um número.")

        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha deve ser finito e não negativo.")

        self.learning_rate = learning_rate
        self.max_iter = int(max_iter)
        self.tol = tol
        self.alpha = alpha

        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None

        self.n_iter_ = 0
        self.converged_ = False

        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Calcula a função sigmoid de forma numericamente estável."""

        result = np.empty_like(z, dtype=np.float64)

        positive = z >= 0
        negative = ~positive

        result[positive] = 1 / (1 + np.exp(-z[positive]))

        exp_z = np.exp(z[negative])

        result[negative] = exp_z / (1 + exp_z)

        return result

    def _loss(self, y: np.ndarray, probabilities: np.ndarray) -> float:
        """Calcula a Binary Cross-Entropy."""

        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

        return -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        """Valida o tipo de X e o converte para float64.

        Usada por fit e predict_proba para que os dois métodos aceitem
        e rejeitem exatamente as mesmas entradas.

        Args:
            X (np.ndarray): Matriz de atributos.

        Returns:
            np.ndarray: Cópia de X com dtype float64.
        """

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X deve ser np.ndarray, recebeu {type(X).__name__}")

        if X.dtype.kind not in "biufO":
            raise ValueError("X deve conter apenas valores numéricos.")

        try:
            return X.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError("X deve conter apenas valores numéricos.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Ajusta o modelo aos dados de treino.

        Args:
            X (np.ndarray): Matriz de atributos.
            y (np.ndarray): Vetor com as classes.

        Returns:
            self: Modelo ajustado.
        """

        X = self._check_X(X)

        if not isinstance(y, np.ndarray):
            raise TypeError(f"y deve ser np.ndarray, recebeu {type(y).__name__}")

        if X.ndim != 2:
            raise ValueError("X deve ser 2D, com shape (n_samples, n_features).")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X não pode ser vazio.")

        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y deve ser 1D e ter o mesmo número de amostras que X.")

        if not np.isfinite(X).all():
            raise ValueError("X contém valores não finitos.")

        # Todo o treino usa variáveis locais. Os atributos do modelo só são atribuídos no final
        n_samples, n_features = X.shape

        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError("LogisticRegression suporta exatamente duas classes.")

        y_binary = (y == classes[1]).astype(np.float64)

        coef = np.zeros(n_features, dtype=np.float64)

        intercept = 0.0

        loss_history = []
        converged = False
        n_iter = 0

        for n_iter in range(1, self.max_iter + 1):
            z = X @ coef + intercept

            probabilities = self._sigmoid(z)

            # Função objetivo J (entropia cruzada + penalidade L2) calculada
            # com os parâmetros anteriores dessa atualização desta iteração.
            loss = self._loss(y_binary, probabilities) + 0.5 * self.alpha * np.sum(coef ** 2)
            loss_history.append(loss)

            error = probabilities - y_binary

            # A penalidade L2 soma alpha * coef ao gradiente dos coeficientes
            # O intercepto não é penalizado
            gradient_coef = X.T @ error / n_samples + self.alpha * coef

            gradient_intercept = np.mean(error)

            # Critério de parada pela norma do gradiente: não depende da
            # learning_rate, ao contrário da mudança nos parâmetros.
            gradient_norm = max(np.linalg.norm(gradient_coef), abs(gradient_intercept))

            if gradient_norm < self.tol:
                converged = True
                break

            coef -= self.learning_rate * gradient_coef
            intercept -= self.learning_rate * gradient_intercept

        self.classes_ = classes
        self.n_features_in_ = n_features
        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.loss_history = loss_history

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calcula as probabilidades previstas para cada classe.

        Args:
            X (np.ndarray): Matriz de atributos.

        Returns:
            np.ndarray: Probabilidades com shape (n_samples, 2), com as
                colunas na ordem de classes_.
        """

        if self.coef_ is None:
            raise RuntimeError("Chame fit antes de predict_proba.")

        X = self._check_X(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X deve ter shape (n_samples, {self.n_features_in_}).")

        if not np.isfinite(X).all():
            raise ValueError("X contém valores não finitos.")

        z = X @ self.coef_ + self.intercept_

        probabilities_class_1 = self._sigmoid(z)
        probabilities_class_0 = 1 - probabilities_class_1

        return np.column_stack((probabilities_class_0, probabilities_class_1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prevê as classes dos dados fornecidos.

        Args:
            X (np.ndarray): Matriz de atributos.

        Returns:
            np.ndarray: Classes previstas.
        """

        if self.coef_ is None:
            raise RuntimeError("Chame fit antes de predict.")

        probabilities = self.predict_proba(X)

        indices = (probabilities[:, 1] >= 0.5).astype(int)

        return self.classes_[indices]