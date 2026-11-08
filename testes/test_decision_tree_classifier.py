import numpy as np
from saikit_learn.arvores_de_decisao.decision_tree_classifier import DecisionTreeClassifier

X = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 12.0], [11.0, 13.0]])

y = np.array([0, 0, 1, 1])

def test_tree_depth():
  """Testa se a profundidade da árvore de decisão está correta."""
  decision_tree_classifier = DecisionTreeClassifier(criterion="gini")
  decision_tree_classifier.fit(X, y)

  depth = decision_tree_classifier.get_depth()
  assert depth <= 2
  
  
  
def test_tree_n_leaves():
  """Testa se a profundidade da árvore de decisão está correta."""
  pass

def test_tree_predict():
  """Testa se a profundidade da árvore de decisão está correta."""
  pass

def test_tree_fit():
  """Testa se a profundidade da árvore de decisão está correta."""
  pass

