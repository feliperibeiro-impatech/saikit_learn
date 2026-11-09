import numpy as np

class PCA:
<<<<<<< HEAD
  def __init__(self, components = 2):
    pass

  def _standardize():
    return 

  def _get_principal_components():
    pass
=======
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
    """Verifica se o PCA já foi ajustado aos dados.

    Raises:
        ValueError: se o PCA não foi ajustado, ou seja, se o método fit ainda não foi executado antes.
    """
    if self.n_components_ is None:
      raise ValueError('O PCA ainda não foi ajustado. Execute o método fit primeiro')
  
  
  def resolve_d(self):
    """Determina o menor número de componentes que atinge a proporção de variância desejada.

    Raises:
        ValueError: se a proporção não pertencer ao intervalo (0, 1].
        ValueError: se a proporção de variância explicada ainda não foi calculada, 
                    ou seja, o método fit ainda não foi executado.

    Returns:
        int: menor número de componentes principais cuja variância explicada acumulada é maior 
        ou igual a 'proporcao'. 
    """
    if not 0 < self.proporcao <= 1:
      raise ValueError('A proporção deve pertencer ao intervalo (0,1]')
    
    if self.explained_variance_ratio_ is None:
      raise ValueError('A proporção de variância explicada ainda não foi calculada, execute o método fit primeiro')
    
    cumulative_sum = np.cumsum(self.explained_variance_ratio_)
    
    return int(np.searchsorted(cumulative_sum, self.proporcao) + 1)
   
  def fit(self, X):
    """Ajusta o PCA aos dados, calculando a média, os autovalores, os componentes principais e a variãncia explicada. 

    Args:
        X (array-like): base de dados com shape (n_amostras, n_atributos).

    Raises:
        ValueError: se 'proporcao' não estiver no intervalo (0, 1].
        ValueError: se 'n_components' não for None nem um inteiro.
        ValueError: se 'n_components' não estiver entre 1 e o número de atributos de 'X'.
        ValueError: se o cálculo da matriz de covariância produzir valores não finitos.
        ValueError: se a variância total dos dados for igual a zero.

    Returns:
        PCA: a própria instância ajustada ('self')
    """
    if not 0 < self.proporcao <= 1:
      raise ValueError('A proporção de variância explicada deve estar no intervalo (0,1]') 
    
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
    
    descending_order = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[descending_order]
    eigen_vectors = eigen_vectors[:, descending_order]
    
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
    self.explained_variance_ = eigen_values[:d]
    self.components_ = eigen_vectors[:, :d]
    
    return self
  
  def transform(self, X):
    """Projeta os dados no espaço dos componentes principais.

    Args:
        X (array-like): base de dados com shape (n_amostras, n_atributos).

    Raises:
        ValueError: se o número de colunas de 'X' for diferente do número de atributos vistos no ajuste.

    Returns:
        NDArray: matriz com shape (n_amostras, n_components_) com os dados centralizados e projetados nos componentes principais.
    """
    self._verify_PCA()
    X = self._valide_matrix(X)
    
    if X.shape[1] != self.n_features_inicial_:
      raise ValueError(f'A matriz dos dados deve possuir {self.n_features_inicial_} colunas.')
    
    X_centered = X - self.mean_
    
    return X_centered @ self.components_
  
  
  def fit_transform(self, X):
    """Ajusta o PCA aos dados e os projeta no espaço das componentes principais;

    Args:
        X (array-like): base de dados com shape (n_amostras, n_atributos).

    Returns:
        NDArray: matriz com shape (n_amostras, n_components_) com os dados centralizados e projetados nos componentes principais.
    """
    return self.fit(X).transform(X)
  
  
  def inverse_transform(self, Z):
    """Reconstrói aproximadamente os dados originais a partir da projeção.

    Args:
        Z (array-like): dados projetados com shape (n_amostras, n_components_).

    Raises:
        ValueError: se o número de colunas de 'Z' for diferente de 'n_components_'

    Returns:
        NDArray: matriz com shape (n_amostras, n_atributos) com os dados reconstruídos no espaço original. 
        A reconstrução é exata apenas se todos os componentes forem mantidos.
    """
    self._verify_PCA()
    Z = self._valide_matrix(Z)
    
    if Z.shape[1] != self.n_components_:
      raise ValueError(f'A matriz deve possuir {self.n_components_} colunas')
    
    return Z @ self.components_.T + self.mean_ 
  
  def plot_eigen_values(self, tol=0.05, show=True ):
    """Plota o scree plot e sugere d pelo método do cotovelo.
    
    O cotovelo é o in[icio do platô: o primeiro componente a partir do qual todas as quedas 
    entre autovalores consecutivos, relativas ao maior autovalor, são menores que 'tol'.

    Args:
        tol (float): limiar de queda relativa que define o platô. Defaults to 0.05.
        show (bool): se True, exibe a figura com plt.show(). Defaults to True.

    Raises:
        ValueError: se 'tol' não pertencer ao intervalo (0, 1).
        ValueError: se os dados tiverem menos de três atributos.

    Returns:
        int: número de componentes sugerido pelo método do cotovelo.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    self._verify_PCA()
    
    if not 0 < tol < 1:
      raise ValueError('A tolerância deve pertencer ao intervalo (0,1)')

    D = self.n_features_inicial_
    if D < 3:
      raise ValueError('Use pelo menos três para analisar o método do cotovelo')
    
    eig = self.eigenvalues_
    
    # queda relativa entre autovalores consecutivos 
    falls = (eig[:-1] - eig[1:]) / eig[0]
    
    # maior queda de i em diante (máximo acumulado de trás para frente)
    maximum_fall = np.maximum.accumulate(falls[::-1])[::-1]
    
    candidates = np.flatnonzero(maximum_fall < tol)
    
    if candidates.size == 0:
      d = D # sem cotovelo claro: mantpem todos os componentes
    else:
      d = max(1, int(candidates[0]))
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, D + 1), eig, marker='o')
    ax.axvline(d, linestyle='--', color='red', label=f'cotovelo: d = {d}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Autovalor')
    ax.set_title('Scree plot')
    ax.legend()
    
    if show:
      plt.show()
    
    return d
>>>>>>> 6ad383e1b8a406395b90c486fd05fb211bf92dd7
