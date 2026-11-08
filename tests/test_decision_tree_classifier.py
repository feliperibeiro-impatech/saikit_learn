import numpy as np

import pytest
from saikit_learn.decision_tree import DecisionTreeClassifier

@pytest.fixture
def dataset():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 12.0], [11.0, 13.0]])
    y = np.array([0, 0, 1, 1])
    return X, y

def test_tree_depth(dataset):
  """Testa se a profundidade da árvore de decisão está correta."""
  X, y = dataset
  decision_tree_classifier = DecisionTreeClassifier(criterion="gini", max_depth=2)
  decision_tree_classifier.fit(X, y)

  assert decision_tree_classifier.depth <= 2
  
def test_tree_n_leaves(dataset):
  r"""Testa se a propriedade n_leaves calcula o número exato de folhas geradas."""
  X, y = dataset
  decision_tree_classifier = DecisionTreeClassifier(criterion="gini")
  decision_tree_classifier.fit(X, y)

  assert decision_tree_classifier.n_leaves == 2

def test_tree_predict(dataset):
  """Testa se a predição é correta."""
  X, y = dataset
  decision_tree_classifier  = DecisionTreeClassifier(criterion="gini")
  decision_tree_classifier.fit(X, y)

  assert decision_tree_classifier.tree is not None
  assert decision_tree_classifier.n_features == 2
  assert np.array_equal(
    decision_tree_classifier.classes, 
    np.array([0, 1])
    )

def test_tree_fit():
  """Testa se a profundidade da árvore de decisão está correta."""
  pass

