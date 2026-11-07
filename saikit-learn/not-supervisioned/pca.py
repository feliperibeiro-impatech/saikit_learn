import numpy as np

class PCA:
  def __init__(self, n_components=None, proporcao=0.95):
    self.n_components = n_components
    self.proporcao = proporcao
    
    self.mean_ = None
    self.n_features_inicial_ = None
    self.n_components_ = None
    self.components_ = None
    self.eigenvalues_ = None
    self.explained_variance_ = None
    self.explained_variance_ratio_ = None
  
  @staticmethod
  def _valide_matrix(X):
    X = np.asarray(X, dtype=float)
    
    if X.ndim != 2:
      raise ValueError('A entrada deve ser uma matriz bidimensional')
    
    if X.shape[0] == 0 or X.shape[1] == 0:
      raise ValueError('A entrada deve ter pelo menos uma linha e uma coluna')
    
    if not np.all(np.isfinite(X)):
      raise ValueError('A entrada não pode conter NaN ou valores infinitos')
    
    return X
  
  def _verify_PCA(self):
    if self.n_components_ is None:
      raise ValueError('O PCA ainda não foi ajustado. Execute o método fit primeiro')
  
  def resolve_d():
    pass
  
  def fit():
    pass
  
  def transform():
    pass
  
  def fit_transform():
    pass
  
  def inverse_transform():
    pass