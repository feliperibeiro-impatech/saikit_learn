import pandas as pd
import numpy as np

class KMeans:
  def __init__(self, n_clusters: int = 2, max_iterations = 100):
    self.max_iterations = max_iterations
    self.n_clusters = n_clusters

  def fit(self, X:np.ndarray) -> None:
    """Executa o algoritmo do KMeans para classificar os pontos em X.
    """
    self.centroids = X[np.random.choice(X.shape[0], self.n_clusters , replace=False)]

    for _ in range(self.max_iterations):
      labels = self._assign_labels(X)

      new_centroids = self._update_centroids(X, labels)

      if np.all(self.centroids == new_centroids):
        break

      self.centroids = new_centroids

  def _assign_labels(self, X: np.ndarray) -> np.ndarray:
    """
      Classifica os pontos de acordo com os pontos que tem a menor distância euclidiana(norma 2) com cada centroide.
    """
    distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis = 2)

    return np.argmin(distances, axis=1)

  def _update_centroids(self, X, labels):
    """
      Atualiza os centroides através das medias dos pontos de um determinado grupo k no intervalo [0, n_clusters - 1].
    """
    new_centroids = np.array([X[labels == i].mean(axis = 0) for i in range(self.n_clusters)])
    return new_centroids

  @property
  def elbow() -> np.ndarray: 
    pass
