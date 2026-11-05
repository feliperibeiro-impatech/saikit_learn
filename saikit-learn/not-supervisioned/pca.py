import numpy as np

class PCA:
  def __init__(self, n_components = None):
    self.n_components = n_components
    self.mean_ = None
    self.n_components = None
    self.explained_variance_ = None
    self.explained_variance_ratio_ = None

  def fit(self, X):
    pass
  
  def transform(self, X):
    pass
  
  def fit_transform(self, X):
    pass
  
  def inverse_transform(self, Z):
    pass
