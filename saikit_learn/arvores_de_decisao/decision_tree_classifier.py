import numpy as np

from typing import Optional, Callable


class TreeNode:
    def __init__(
        self,
        feature: Optional[int] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[float] = None,
        gain: Optional[float] = None,
        threshold: Optional[float] = None,
        n_samples: int = 0,
    ):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain
        self.threshold = threshold
        self.n_samples = n_samples

    @property
    def is_leaf(self) -> None:
        return self.feature is None


class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "gini",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain: float = 0.0,
        max_depth: Optional[int] = None,
        split_strategy: str = "all",
    ):
        if criterion not in ("gini", "entropy", "classification_error"):
            raise ValueError(
                f"O critério {criterion!r} não existe, por favor insira um critério válido"
            )
        if split_strategy not in ("all", "median"):
            raise ValueError(
                f"A estratégia de corte {split_strategy} não existe, por favor insira uma estratégia de corte válida."
            )

        self.criterion = criterion
        self.min_samples = min_samples_split
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.split_strategy = split_strategy
        self.min_sample_leafs = min_samples_leaf

        self.tree: Optional[TreeNode] = None
        self.classes: Optional[np.ndarray] = None
        self.n_features: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes, y_index = np.unique(y, return_inverse=True)
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y_index, depth=0)

        return self

    def _get_best_split(self, X: np.ndarray, y: np.ndarray):
        """ """
        n, n_features = X.shape
        n_classes = len(self.classes)
        impurity = self._get_criterion_function()
        onehot = np.eye(n_classes)[y]

        best = None
        best_score = np.inf

        for j in range(n_features):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]

            cum = np.vstack(
                [np.zeros((1, n_classes)), np.cumsum(onehot[order], axis=0)]
            )

            thresholds = self._candidate_thresholds(xs)
            if thresholds.size == 0:
                continue

            n_left = np.searchsorted(xs, thresholds, side="right")
            n_right = n - n_left

            valid = (n_left >= self.min_samples_leaf) & (
                n_right >= self.min_samples_leaf
            )
            valid &= (n_left > 0) & (n_right > 0)

            thresholds, n_left, n_right = (
                thresholds[valid],
                n_left[valid],
                n_right[valid],
            )

            left_counts = cum[n_left]
            right_counts = cum[n] - left_counts
            q_left = impurity(left_counts / n_left[:, None])
            q_right = impurity(right_counts / n_right[:, None])
            scores = ((n_left * q_left) + (n_right * q_right)) / n

            i = int(np.argmin(scores))
            if scores[i] < best_score:
                best_score = float(scores[i])
                best = (j, float(thresholds[i]), best_score)

        return best

    def _get_criterion_function(self) -> Callable[[np.ndarray], np.ndarray]:
        match self.criterion:
            case "gini":
                return self._gini
            case "classification_error":
                return self._classification_error
            case "entropy":
                return self._entropy
            case _:
                raise ValueError(f"Critério: {self.criterion!r} não existente.")

    def train(self, X: np.array, Y: np.array) -> None:
        """Treina os dados X para nos dados Y.

        Args:
            X (np.ndarray): Dados de entrada.
            Y (np.ndarray): Valores observados.

        Returns:
            decision_tree (DecisionTreeClassifier):
        """
        return self.fit(X, Y)

    def _p(k: int, y: np.ndarray):
        """
        Probabilidade de um certo nó m ser classificado como k, (p_mk).
        """
        return float(np.mean(y == k))

    @staticmethod
    def _classification_error(p: np.ndarray) -> np.ndarray:
        return 1 - np.max(p, axis=1)

    @staticmethod
    def _entropy(p: np.ndarray) -> np.ndarray:
        safe = np.where(p > 0, p, 1.0)
        return -np.sum(p * np.log2(safe), axis=1)

    @staticmethod
    def _gini(p: np.ndarray) -> np.ndarray:
        return np.sum(p * (1 - p), axis=1)

    def _class_probabilities(self, y: np.ndarray) -> np.ndarray:
        n_classes = len(self._classes)

        return np.array([self._p(k, y) for k in range(n_classes)])

    def _check_fitted(self) -> None:
        if self.tree is None:
            raise ("Árvore não treinada, por favor utilize o método fit antes")

    def get_n_leaves(self) -> int:
        self._check_fitted()

        def rec(node: TreeNode) -> int:
            return 1 if node.is_leaf else rec(node.left) + rec(node.right)

        return rec(self.tree)

    def get_depth(self) -> int:
        self._check_fitted()

        def rec(node: TreeNode):
            return 0 if node.is_leaf else 1 + max(rec(node.left), rec(node.right))

        return rec(self.tree)

    def _predict_one(self, x: np.ndarray) -> np.ndarray:
        node = self.tree
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right

        return node.value

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        r"""Vetor de todas a probabilidades p.

        Args:
            p (np.ndarray): Vetor com a proporção de cada classe :math:`\hat{p}_{mk}`.

        Returns:
            p_mk (np.ndarray): Array de vetores p_mk
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return np.array([self._predict_one(x) for x in X])

    def predict(self, X: np.ndarray, y: np.ndarray) -> float:
        """Prevê o R_m em que o x está situado.

        Args:
            X (np.ndarray): Dados de entrada.
            y (np.ndarray): Valores observados.

        Returns:
            classifications: Classificações do modelo
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mede a acurácia do modelo observando quantos valores numa escala de 0.0 à 1.0.

        Args:
            X (np.ndarray): Dados de entrada.
            y (np.ndarray): Valores observados.

        Returns:
            score (float): Acertos médios doo classificador y.
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))
