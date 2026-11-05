import numpy as np

class PCA:
  def __init__(self, n_components = None):
    self.n_components = n_components
  

  def fit(self, X):
    # converte os dados de entrada em um array NumPy e resgata o número de amostras e o número de variáveis
    X = np.asarray(X, dtype=float)
    N, D = X.shape  
    
    # Calcula a média das amostras e centraliza X em relação a essa média 
    self.mean_ = X.mean(axis=0)  
    Xc = X - self.mean_  
    
    # Constrói a matriz de covariâncias
    S = (Xc.T @ Xc) / N  
    
    # Calcula os autovalores e autovetores da matriz de covariância por meio da função np.linalg.eigh.
    eigen_values, eigen_vectors = np.linalg.eigh(S)
    
    # Reordena os autovalores em ordem decrescente (pois a função np.linalg.eigh retorna, por padrão,
    # os autovalores em ordem crescente) e ordena também os autovetores correspondentes a eles
    order = np.argsort(eigen_values)[::-1]
    eigen_values = np.clip(eigen_values[order], 0.0, None)  # garante que todos os autovalores são maiores ou iguais a 0
    eigen_vectors = eigen_vectors[:, order]
    
    
    
    return 
  
  def transform(self, X):
    pass
  
  def fit_transform(self, X):
    pass
  
  def inverse_transform(self, Z):
    pass
