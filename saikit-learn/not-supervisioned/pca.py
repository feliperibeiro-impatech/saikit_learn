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
    
     
    
    
    
    return 
  
  def transform(self, X):
    pass
  
  def fit_transform(self, X):
    pass
  
  def inverse_transform(self, Z):
    pass
