import numpy as np

from saikit_learn.nao_supervisionado.k_means import KMeans

X = np.array([
    (1, 1),
    (3, 1),
    (2, 3),
    (12, 13),
    (12, 15),
    (15, 15),
  ]) 

def test_kmeans_centroids():
  """
    Teste para verificar se os centroides com n_clusters = 2 estão corretos.  
  """
  k_means = KMeans(n_clusters = 2)
  k_means.fit(X)

  cond1 = (2, 2.5) in k_means.centroids
  cond2 = (13, 11) in k_means.centroids 

  assert (cond1 and cond2) == True

def test_kmeans_labels():
  """
    Teste para averiguar se os 'labels' estão corretamente classificados. Verifica se:
      * As linhas 0 e 1 possuem o mesmo rótulo;
      * As linhas 0 e 3 possuem rótulos diferentes.
  """
  np.random.seed(42)
  k_means = KMeans(n_clusters=2)
  k_means.fit(X)
  labels = k_means._assign_labels(X)

  assert (labels[0] == 0) and (labels[3] == 1)
  assert (labels[0] == labels[1]) and (labels[0] != labels[3])

def test_kmeans_elbow_k_value():
  """
    Testa se o método do cotovelo alcança o valor ótimo de n_clusters.
  """
  np.random.seed(42)
  k_means = KMeans(n_clusters=3)
  best_k = k_means.find_best_k_value(X, 5, method="elbow")

  assert best_k == 2

def test_kmeans_silhouette_score():
  """
    Testa se o método com silhouette score alcança o valor ótimo de n_clusters. 
  """
  k_means = KMeans(n_clusters=3)
  best_k = k_means.find_best_k_value(X, 5, method="silhouette")

  assert best_k == 2
  