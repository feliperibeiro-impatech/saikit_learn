import numpy as np

from ._tree_node import TreeNode
from typing import Optional


class DecisionTreeRegressor:
    def __init__(
        self,
        criterion: str = "squared_error",
        min_samples_leaf: int = 1,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_gain: float = 0.0,
        split_strategy: str = "all",
    ):
        if criterion != "squared_error":
            raise ValueError(f"Critério inválido {criterion!r}")
        if split_strategy not in ("all", "median"):
            raise ValueError(f"Estratégia de corte inválida {split_strategy!r}")

        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.split_strategy = split_strategy

        self.tree: Optional["TreeNode"] = None
        self.n_features: Optional[int] = None

    def train(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        return self.fit(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("as amostras X deve ser em duas dimensões (n_amostras, n_features)")
        if y.ndim != 1:
            raise ValueError("y deve ser 1D e ter o mesmo número de linhas que X")

        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n = len(y)
        node_value = float(np.mean(y))
        node = TreeNode(value=node_value, n_samples=n)
        node_impurity = self._impurity(y)

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if n < self.min_samples_split:
            return node
        if node_impurity <= 1e-12:
            return node

        split = self._get_best_split(X, y)
        if split is None:
            return node

        feature, threshold, children_impurity = split
        gain = node_impurity - children_impurity
        if gain < self.min_gain:
            return node

        node.feature = feature
        node.threshold = threshold
        node.gain = gain

        mask = X[:, feature] <= threshold
        node.left = self._build_tree(X[mask], y[mask], depth=depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth=depth + 1)
        return node

    @staticmethod
    def _impurity(y: np.ndarray) -> float:
        """Q(R) = (1/n) Σ (y_i - ȳ)²  (variância dentro do nó).

        Args:
            `y` (np.ndarray): Dados observados

        Returns:
            `impurity_value` (float): Valor da impureza.
        """
        return float(np.mean((y - np.mean(y))) ** 2)

    def _get_best_split(self, X: np.ndarray, y: np.ndarray):
        """Escolhe os melhores valores de corte de `X` para `y`.

        Args:
            `X` (np.ndarray): Dados de entrada.
            `y` (np.ndarray): Dados observados.

        Returns:
            `best_split` (tuple(int, float, float)): Vetor dos melhores valores de corte.
        """
        n, n_features = X.shape
        y_c = y - np.mean(y)

        best = None
        best_score = np.inf

        for j in range(n_features):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            ys = y_c[order]

            cum_y = np.concatenate([[0.0, 0.0], np.cumsum(ys)])
            cum_y2 = np.concatenate([[0.0, 0.0], np.cumsum(ys**2)])

            thresholds = self._get_candidate_thresholds(xs)
            if thresholds.size == 0:
                continue

            n_left = np.searchsorted(xs, thresholds, side="right")
            n_right = n - n_left

            valid = (n_left >= self.min_samples_leaf) & (
                n_right >= self.min_samples_leaf
            )
            valid &= (n_left > 0) & (n_right > 0)

            if not valid.any():
                continue

            thresholds, n_left, n_right = (
                thresholds[valid],
                n_left[valid],
                n_right[valid],
            )

            sum_l, sum_l2 = cum_y[n_left], cum_y2[n_left]
            sum_r, sum_r2 = cum_y[n] - sum_l, cum_y2[n] - sum_l2

            rss_left = np.maximum(sum_l2 - sum_l**2 / n_left, 0.0)
            rss_right = np.maximum(sum_r2 - sum_r**2 / n_right, 0.0)
            scores = (rss_left + rss_right) / n

            i = int(np.argmin(scores))
            if scores[i] < best_score:
                best_score = float(scores[i])
                best = (j, float(thresholds[i]), best_score)

        return best

    def _get_candidate_thresholds(self, xs_sorted: np.ndarray) -> np.ndarray:
        """Seleciona os candidatos ao corte `s` da árvore de decisão.

        Args:
            `xs_sorted` (np.ndarray): Pontos x ordenados de forma crescente.

        Returns:
            `candidate_thresholds` (np.ndarray): Possíveis cortes.
        """
        if self.split_strategy == "median":
            u = np.unique(xs_sorted)

            if u.size == 0:
                return np.empty(0)

            index = len(u) / 2
            return np.array([(u[index - 1] + u[index]) / 2])

        u = np.unique(xs_sorted)
        if u.size < 2:
            return np.empty(0)

        mid = (u[:-1] + u[1:]) / 2
        return np.where(mid >= u[1:], u[:-1], mid)

    def _predict_one(self, x: np.ndarray) -> np.ndarray:
        node = self.tree
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right

        return node.value

    def predict(self, X: np.ndarray) -> float:
        """Prevê o R_m em que o x está situado.

        Args:
            `X` (np.ndarray): Dados de entrada.
            `y` (np.ndarray): Valores observados.

        Returns:
            `classifications`: Classificações do modelo
        """
        self._check_fitted()

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X.reshape(1, -1)

        return np.array([self._predict_one(x) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcula o score alcançado pela regressão

        Args:
            `X` (np.ndarray): Dados de entrada.
            `y` (np.ndarray): Dados de saída.

        Returns:
            `score` (float): Valor do score
        """
        y = np.asarray(y, dtype=float)
        rss = np.sum((y - self.predict(X)) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)

        return float(1 - rss / tss) if tss > 0 else 0.0

    def _check_fitted(self) -> None:
        if self.tree is None:
            raise ValueError("Árvore não treinada, chame `fit(X, y)` antes.")

    @property
    def n_leaves(self) -> int:
        """Obtém a quantidade de folhas na árvore de decisão.

        Returns:
            `n_leaves` (int): Quantidade de folhas
        """
        self._check_fitted()

        def rec(node: TreeNode) -> int:
            return 1 if node.is_leaf else rec(node.left) + rec(node.right)

        return rec(self.tree)

    @property
    def depth(self) -> int:
        """Obtém a profundidade da árvore de decisão.

        Returns:
            depth: Profundidade da árvore de decisão.
        """
        self._check_fitted()
 
        def rec(node: TreeNode):
            return 0 if node.is_leaf else 1 + max(rec(node.left), rec(node.right))

        return rec(self.tree)
