import numpy as np
import pytest
from saikit_learn.decision_tree import DecisionTreeClassifier


@pytest.fixture
def dataset():
  """Dataset 2D linearmente separável para testes da árvore de decisão."""
  X = np.array([[2.0, 2.0], [2.0, 3.0], [10.0, 12.0], [11.0, 13.0]])
  y = np.array([0, 0, 1, 1])
  return X, y

def test_init_invalid_parameters():
  """Garante que a classe levanta ValueError para critérios ou estratégias de corte inválidas."""
  with pytest.raises(ValueError):
    DecisionTreeClassifier(criterion="invalid_criterion")

  with pytest.raises(ValueError):
    DecisionTreeClassifier(split_strategy="invalid_strategy")

def test_check_fitted_exceptions(dataset):
  """Testa se acessar propriedades ou predições antes do fit gera exceção."""
  X, _ = dataset
  clf = DecisionTreeClassifier()

  with pytest.raises(Exception):
    _ = clf.depth

  with pytest.raises(Exception):
    _ = clf.n_leaves

  with pytest.raises(Exception):
    clf.predict(X)

  with pytest.raises(Exception):
    clf.predict_proba(X)

def test_impurity_criteria_math():
  """Verifica a exatidão matemática das funções estáticas de impureza."""
  pure_p = np.array([[1.0, 0.0]])
  assert DecisionTreeClassifier._gini(pure_p)[0] == pytest.approx(0.0)
  assert DecisionTreeClassifier._entropy(pure_p)[0] == pytest.approx(0.0)
  assert DecisionTreeClassifier._classification_error(pure_p)[
      0
  ] == pytest.approx(0.0)

  mixed_p = np.array([[0.5, 0.5]])
  assert DecisionTreeClassifier._gini(mixed_p)[0] == pytest.approx(0.5)
  assert DecisionTreeClassifier._entropy(mixed_p)[0] == pytest.approx(1.0)
  assert DecisionTreeClassifier._classification_error(mixed_p)[
      0
  ] == pytest.approx(0.5)

def test_tree_depth(dataset):
  """Testa se a propriedade depth calcula a profundidade e respeita max_depth."""
  X, y = dataset
  clf = DecisionTreeClassifier(criterion="gini", max_depth=2)
  clf.fit(X, y)

  assert clf.depth <= 2

def test_tree_n_leaves(dataset):
  """Testa se a propriedade n_leaves calcula o número exato de folhas geradas."""
  X, y = dataset
  clf = DecisionTreeClassifier(criterion="gini")
  clf.fit(X, y)

  assert clf.n_leaves == 2

def test_min_samples_split(dataset):
  """Verifica se min_samples_split alto impede novos splits na raiz."""
  X, y = dataset
  clf = DecisionTreeClassifier(min_samples_split=10)
  clf.fit(X, y)

  assert clf.depth == 0
  assert clf.n_leaves == 1

def test_tree_fit(dataset):
  """Testa se o método fit ajusta os atributos internos da árvore."""
  X, y = dataset
  clf = DecisionTreeClassifier(criterion="gini")
  clf.fit(X, y)

  assert clf.tree is not None
  assert clf.n_features == 2
  assert np.array_equal(clf.classes, np.array([0, 1]))

def test_tree_predict(dataset):
  """Testa a acurácia das predições, o score e o cálculo de probabilidades."""
  X, y = dataset
  clf = DecisionTreeClassifier(criterion="gini")
  clf.fit(X, y)

  y_pred = clf.predict(X)
  assert np.array_equal(y_pred, y)
  assert clf.score(X, y) == 1.0

  X_new = np.array([[1.5, 2.5], [10.5, 12.5]])
  y_new = np.array([0, 1])
  assert np.array_equal(clf.predict(X_new), y_new)

  probabilities = clf.predict_proba(X_new)
  assert probabilities.shape == (2, 2)
  assert np.allclose(probabilities[0], [1.0, 0.0])
  assert np.allclose(probabilities[1], [0.0, 1.0])