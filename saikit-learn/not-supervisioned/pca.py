import numpy as np

class PCA:
  """Análise dos Componentes Principais por meio da Matriz de Covariância
  """
  
  def __init__(self, n_components=None, proporcao=0.95):
    """Inicializa as instâncias do objeto da classe

    Args:
        n_components (int): número de componentes que se deseja projetar os dados. Defaults to None.
        proporcao (float): proporção de variância expicada que se deseja usar. Defaults to 0.95.
    """
    
    # Parâmetros definidos pelo usuário 
    self.n_components = n_components
    self.proporcao = proporcao
    
    # Parâmetros aprendidos 
    self.mean_ = None  # média dos dados
    self.n_features_inicial_ = None  # dimensão inicial (D) 
    self.n_components_ = None  # número de componentes determinada pelo modelo (d)
    self.components_ = None  # componentes determinadas pela decomposição da matriz 
    self.eigenvalues_ = None  # autovalores da matriz de covariancia 
    self.explained_variance_ = None # variância explicada 
    self.explained_variance_ratio_ = None  # proporção de variância explicada
  
  @staticmethod
  def _valide_matrix(X):
    """Converte a entrada em um array NumPy e verifica o formato e os valores presentes.

    Args:
        X (array-like): base de dados com shape (n_amostras, n_atriutos).

    Raises:
        ValueError: se a entrada não for bidimensional.
        ValueError: se a entrada não tiver pelo menos uma linha e uma coluna.
        ValueError: se a entrada contiver NaN ou valores infinitos.

    Returns:
        NDArray: matriz de dados convertida em array NumPy (dtype float).
    """
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
  
  def resolve_d(self):
    if not 0 < self.proporcao <= 1:
      raise ValueError('A proporção deve pertencer ao intervalo (0,1]')
    
    if self.explained_variance_ratio_ is None:
      raise ValueError('A proporção de variância explicada ainda não foi calculada, execute o método fit primeiro')
    
    cumulative_sum = np.cumsum(self.explained_variance_ratio_)
    
    return np.searchsorted(cumulative_sum, self.proporcao) + 1
   
  def fit(self, X):
    X = self._valide_matrix(X)
    N, D = X.shape
    
    if self.n_components is not None:
      if isinstance(self.n_components, (bool,np.bool_)) or not isinstance(self.n_components, (int, np.integer)):
        raise ValueError('O número de componentes deve ser um inteiro ou o atributo booleano None')
      
      if not 1 <= self.n_components <= D:
        raise ValueError(f'O número de componentes deve estar entre 1 e {D}')
      
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    S = (X_centered.T @ X_centered) / N
    
    if not np.all(np.isfinite(S)):
      raise ValueError('O cálculo da matriz de covariâncias produziu valores não finitos')
    
    eigen_values, eigen_vectors = np.linalg.eigh(S)
    
    order = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[order]
    eigen_vectors = eigen_vectors[:, order]
    
    total_variance = np.sum(eigen_values)
    
    if total_variance == 0:
      raise ValueError('Os dados possuem variância total igual a zero. A proporção de variância explicada é indefinida.')
    
    self.explained_variance_ratio_ = (eigen_values / total_variance)
    
    if self.n_components is None:
      d = self.resolve_d()
    else:
      d = self.n_components
    
    self.mean_ = mean
    self.n_features_inicial_ = D
    self.eigenvalues_ = eigen_values
    self.n_components_ = d
    self.explained_variance_ = eigen_values[:, :d]
    self.components_ = eigen_vectors[:, :d]
    
    return self
  
  def transform(self):
    self._verify_PCA()
    X = self._valide_matrix(X)
    
    if X.shape[1] != self.n_features_inicial_:
      raise ValueError(f'A matriz dos dados deve possui{self.n_features_inicial_} colunas.')
    
    X_centered = X - self.mean_
    
    return X_centered @ self.components_
  
  
  def fit_transform(self, X):
    return self.fit(X).transform(X)
  
  
  def inverse_transform(self, Z):
    self._verify_PCA()
    Z = self._valide_matrix(Z)
    
    if Z.shape[1] != self.n_components_:
      raise ValueError(f'A matriz deve possuir {self.n_components_} colunas')
    
    return Z @ self.n_components_.T + self.mean_