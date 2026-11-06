import numpy as np

class PCA:
  def __init__(self, n_components = None):
    self.n_components = n_components
    self.mean_ = None
    self.eigenvalues_ = None
    self.explained_variance_ratio_ = None

  def fit(self, X):
    
    # Transforma a fonte dos dados em um array NumPy e resgata o número de amostras e o núumero de features
    X = np.asarray(X, dtype=float)
    N, D = X.shape
     
    # Centraliza os dados em relação à média das amostras
    self.mean_ = np.mean(X, axis=0)
    X_centered = X - self.mean_
    
    # Constrói matriz de covariância S = (1/N) * X_centerd^T * X_centered
    S = np.cov(X_centered, rowvar=False, bias=True)
    
    # Autodecomposição da matriz S (Usamos eigh pois S é simétrica e semidefinida positiva)
    eigen_values, eigen_vectors = np.linalg.eigh(S)
    
    # Reordena os autovalores em ordem decrescente
    order = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[order]
    eigen_vectors = eigen_vectors[:, order]
     
    pass
  
  def transform(self, X):
    pass
  
  def fit_transform(self, X):
    pass
  
  def inverse_transform(self, Z):
    pass
