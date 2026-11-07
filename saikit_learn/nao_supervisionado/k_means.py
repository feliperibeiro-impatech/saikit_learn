import numpy as np
from typing import Optional


class KMeans:
    def __init__(self, n_clusters: Optional[int], max_iterations=100):
        """
        Args:
          n_clusters (int): Número de Clusters (hiperparâmetro).
          max_interation (int): Número de iterações máximo que o algoritmo fará para obter os centróides.
        """
        self.max_iterations = max_iterations
        self.n_clusters = n_clusters
        self._inertia = None
        self.centroids = None

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Executa o algoritmo do KMeans para classificar os pontos em X.

        Args:
          X: Dados de entrada.

        Returns:
          Kmeans
        """
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iterations):
            labels = self._assign_labels(X)

            new_centroids = self._update_centroids(X, labels)

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        final_labels = self._assign_labels(X)
        self._inertia = self._calculate_inertia(X, final_labels)

        return self

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Classifica os pontos de acordo com os pontos que tem a menor distância euclidiana(norma 2) com cada centroide.

        Args:
          X: Dados de entrada
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        return np.argmin(distances, axis=1)

    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        distances = np.linalg.norm(X - self.centroids[labels], axis=1)

        return float(np.sum(distances**2))

    def _calculate_silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        n_samples = X.shape[0]
        unique_labels = np.unique(labels)

        dist_matrix = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)

        a = np.zeros(n_samples)
        b = np.full(n_samples, np.inf)
        is_single_cluster = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            own_label = labels[i]

            same_cluster_mask = (labels == own_label) & (np.arange(n_samples) != i)

            if not np.any(same_cluster_mask):
                is_single_cluster[i] = True
                continue

            if np.any(same_cluster_mask):
                a[i] = np.mean(dist_matrix[i, same_cluster_mask])
            else:
                a[i] = 0.0

            for label in unique_labels:
                if label == own_label:
                    continue

                other_cluster_mask = labels == label
                other_dist_mean = np.mean(dist_matrix[i, other_cluster_mask])

                if other_dist_mean < b[i]:
                    b[i] = other_dist_mean

        denom = np.maximum(a, b)
        s = np.zeros_like(denom)

        valid_mask = np.isfinite(denom) & (denom > 0)
        np.divide(b - a, denom, out=s, where=valid_mask)

        return float(np.mean(s))

    @classmethod
    def find_best_k_value(
        cls: "KMeans", X: np.ndarray, max_k: int = 10, method="silhouette"
    ) -> int:
        match method:
            case "silhouette":
                k_values = list(range(2, max_k + 1))
                best_k = k_values[0]
                max_score = -1.0

                for k in k_values:
                    model = cls(n_clusters=k)
                    model.fit(X)
                    labels = model._assign_labels(X)
                    score = model._calculate_silhouette_score(X, labels)

                    if score > max_score:
                        max_score = score
                        best_k = k

                return best_k

            case "elbow":
                all_k = list(range(1, max_k + 1))
                inertias = []

                for k in all_k:
                    model = cls(n_clusters=k).fit(X)
                    inertias.append(model._inertia)

                return cls.find_best_k_elbow(all_k, inertias)

    @classmethod
    def _elbow(cls, X: np.ndarray, max_k: int = 10) -> dict[str, list]:
        """
        Calcula todos os pontos (k, inertia) de 1 até max_k

        Args:
          k: Dados de entrada
          max_k: Quantidade máxima de clusters.

        Returns:
          dict[str, list]: Objeto contendo todos os clusters e suas inércias respectivamente.
        """
        k_values = list(range(1, max_k + 1))
        inertias = []

        for k in k_values:
            model = cls(n_clusters=k)
            model.fit(X)
            inertias.append(model._inertia)

        return {"k_values": k_values, "inertias": inertias}

    def find_best_k_elbow(k_values: list[int], inertias: list[float]) -> int:
        """
        Encontra o ponto (k, inertia) com maior distância perpendicular dentre os pontos(método do cotovelo).
        """
        k_norm = (np.array(k_values) - min(k_values)) / (max(k_values) - min(k_values))
        inertia_norm = (np.array(inertias) - min(inertias)) / (
            max(inertias) - min(inertias)
        )

        p1 = np.array([k_norm[0], inertia_norm[0]])
        pm = np.array([k_norm[-1], inertia_norm[-1]])

        v = pm - p1

        points = np.column_stack((k_norm, inertia_norm))
        w = points - p1

        distances = []

        cross_2d = v[0] * w[:, 1] - v[1] * w[:, 0]
        distances = np.abs(cross_2d) / np.linalg.norm(v)

        best_index = np.argmax(distances)
        return k_values[best_index]

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Atualiza os centroides através das medias dos pontos de um determinado grupo k no intervalo [0, n_clusters - 1].
        """
        new_centroids = np.array(
            [X[labels == i].mean(axis=0) for i in range(self.n_clusters)]
        )

        return new_centroids
