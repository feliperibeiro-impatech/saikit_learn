import numpy as np

class PCA:
  def __init__(self, n_components = None):
    self.n_components = n_components
    self.mean_ = None
    self.n_components = None
    self.eigenvalues_ = None
    self.explained_variance_ratio_ = None

  def fit(self, X):
    
    # Transforma a fonte dos dados em um array NumPy e resgata o número de amostras e o núumero de features
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    
    # Se n_components não for especificado, mantém todas as D dimensões originais
    if self.n_components is None:
      self.n_components = D
    
    # Centraliza os dados em relação à média das amostras
    self.mean_ = np.mean(X, axis=0)
    X_centered = X - self.mean_
    
    pass
  
  def transform(self, X):
    pass
  
  def fit_transform(self, X):
    pass
  
  def inverse_transform(self, Z):
    pass
