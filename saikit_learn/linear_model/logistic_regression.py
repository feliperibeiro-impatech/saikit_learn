import numpy as np


class LogisticRegression:
    """Regressão Logística treinada por Gradient Descent.

    Com duas classes, treina um único classificador, que estima a probabilidade
    da classe positiva (classes_[1]). Com mais de duas classes, treina um classificador 
    binário por classe, que distingue essa classe de todas as demais (One Vs Rest).
    Como o Gradient Descent é sensível à escala dos atributos, padronize X antes do fit

    Cada classificador minimiza a Binary Cross-Entropy média, com penalidade
    L2 opcional nos coeficientes (o intercepto não é penalizado):

        J(w, b) = BCE(w, b) + (alpha / 2) * ||w||^2

    No caso de ter mais de 2 classes, as probabilidades são os scores 
    dos classificadores normalizados para somar 1. 

    Attributes:
        classes_ (np.ndarray): Classes vistas no treino, em ordem crescente.
        coef_ (np.ndarray): Coeficientes dos classificadores, com shape
            (n_classifiers, n_features). Cada linha é um classificador:
            n_classifiers é 1 com duas classes e n_classes com mais de duas.
        intercept_ (np.ndarray): Interceptos, com shape (n_classifiers,).
        n_features_in_ (int): Número de atributos vistos no fit.
        n_iter_ (np.ndarray): Iterações realizadas por cada classificador.
        converged_ (np.ndarray): Indica se cada classificador atingiu o
            critério de parada pelo gradiente antes de max_iter. Use
            converged_.all() para saber se todos convergiram.
        loss_history (list[list[float]]): Valor de J em cada iteração, uma
            lista por classificador, calculado com os parâmetros anteriores
            à atualização daquela iteração.
    """

    def __init__(self, learning_rate: float = 0.5, max_iter: int = 10000, tol: float = 1e-6, alpha: float = 0.0):
        """Inicializa a Regressão Logística.

        Args:
            learning_rate (float): Taxa de aprendizado do Gradient Descent.
            max_iter (int): Número máximo de iterações de cada classificador.
            tol (float): Tolerância para o critério de parada. O treino de
                cada classificador termina quando a norma do gradiente fica
                abaixo de tol.
            alpha (float): Força da regularização L2. Com 0.0 (padrão), não há
                regularização. Quanto maior o alpha, menor deve ser a
                learning_rate para o treino não divergir.
        """

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
        self.n_iter_ = None
        self.converged_ = None
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
        """Calcula a Binary Cross-Entropy média."""

        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

        return -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        """Valida o tipo de X e o converte para float64.

        Usada por fit e predict_proba para que os dois métodos aceitem e rejeitem
        exatamente as mesmas entradas.

        Args:
            X (np.ndarray): Matriz de atributos.

        Returns:
            np.ndarray: X convertido para dtype float64.
        """

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X deve ser np.ndarray, recebeu {type(X).__name__}")

        if X.dtype.kind not in "biufO":
            raise ValueError("X deve conter apenas valores numéricos.")

        try:
            return X.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError("X deve conter apenas valores numéricos.") from None

    def _fit_binary(self, X: np.ndarray, y: np.ndarray):
        """Treina um classificador binário por Gradient Descent.

        Args:
            X (np.ndarray): Matriz de atributos, já validada.
            y (np.ndarray): Vetor com 0.0 e 1.0 (1.0 é a classe positiva).

        Returns:
            tuple: (coef, intercept, n_iter, converged, loss_history).
        """

        n_samples, n_features = X.shape

        coef = np.zeros(n_features, dtype=np.float64)
        intercept = 0.0

        loss_history = []
        converged = False
        n_iter = 0

        for n_iter in range(1, self.max_iter + 1):
            z = X @ coef + intercept
            probabilities = self._sigmoid(z)

            # Função objetivo J (entropia cruzada + penalidade L2) calculada
            # com os parâmetros anteriores à atualização desta iteração.
            loss = self._loss(y, probabilities) + 0.5 * self.alpha * np.sum(coef ** 2)
            loss_history.append(loss)

            error = probabilities - y

            # A penalidade L2 soma alpha * coef ao gradiente dos
            # coeficientes. O intercepto não é penalizado.
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

        return coef, intercept, n_iter, converged, loss_history

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Ajusta o modelo aos dados de treino.

        Com duas classes, treina um classificador para a classe positiva
        (classes_[1]). Com mais de duas, treina um por classe (One-vs-Rest).

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

        if y.dtype.kind == "f" and not np.isfinite(y).all():
            raise ValueError("y contém valores não finitos.")

        # Todo o treino usa variáveis locais. Os atributos do modelo só
        # são atribuídos no final, para que um fit que falha não deixe
        # o modelo num estado inconsistente.
        n_features = X.shape[1]
        classes = np.unique(y)

        if len(classes) < 2:
            raise ValueError("LogisticRegression requer pelo menos duas classes.")

        # Com duas classes, os dois classificadores OvR seriam espelhos um do
        # outro, então basta treinar o da classe positiva.
        targets = classes[1:] if len(classes) == 2 else classes
        n_classifiers = len(targets)

        coef = np.zeros((n_classifiers, n_features), dtype=np.float64)
        intercept = np.zeros(n_classifiers, dtype=np.float64)
        n_iter = np.zeros(n_classifiers, dtype=int)
        converged = np.zeros(n_classifiers, dtype=bool)
        loss_history = []

        for i, current_class in enumerate(targets):
            y_binary = (y == current_class).astype(np.float64)

            coef[i], intercept[i], n_iter[i], converged[i], current_loss_history = self._fit_binary(X, y_binary)
            loss_history.append(current_loss_history)

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

        Com duas classes, as probabilidades vêm do classificador da classe
        positiva. Com mais de duas, são os scores dos classificadores
        One-vs-Rest normalizados para que cada linha some 1.

        Args:
            X (np.ndarray): Matriz de atributos. Um vetor 1D é tratado como
                uma única amostra.

        Returns:
            np.ndarray: Probabilidades com shape (n_samples, n_classes), com
                as colunas na ordem de classes_.
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

        z = X @ self.coef_.T + self.intercept_

        if len(self.classes_) == 2:
            probabilities_class_1 = self._sigmoid(z[:, 0])

            return np.column_stack((1 - probabilities_class_1, probabilities_class_1))

        # Normaliza os scores sigmoid em escala log. Dividir os sigmoids
        # diretamente daria 0 / 0 quando todos fossem arredondados para 0.
        log_scores = -np.logaddexp(0, -z)
        log_scores -= log_scores.max(axis=1, keepdims=True)

        probabilities = np.exp(log_scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prevê as classes dos dados fornecidos.

        Com duas classes, prevê a positiva se sua probabilidade for >= 0.5.
        Com mais de duas, prevê a de maior probabilidade.

        Args:
            X (np.ndarray): Matriz de atributos.

        Returns:
            np.ndarray: Classes previstas.
        """

        if self.coef_ is None:
            raise RuntimeError("Chame fit antes de predict.")

        probabilities = self.predict_proba(X)

        if len(self.classes_) == 2:
            indices = (probabilities[:, 1] >= 0.5).astype(int)
        else:
            indices = np.argmax(probabilities, axis=1)

        return self.classes_[indices]