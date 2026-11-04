import numpy as np

class PCA:
  def __init__(self, n_components = None):
    self.n_components = n_components
  

  def fit(self, X):
    X = np.asarray(X, dtype=float)  # converte os dados de entrada em um array NumPy
    N, D = X.shape  # resgata o número de amostras e o número de variáveis
     
    self.mean_ = X.mean(axis=0)  # calcula a média das amostras
    Xc = X - self.mean_  # centraliza X em relação à média
    
    S = (Xc.T @ Xc) / N  # constrói a matriz de covariâncias
    return 
  
  def transform(self, X):
    pass
  
  def fit_transform(self, X):
    pass
  
  def inverse_transform(self, Z):
    pass
