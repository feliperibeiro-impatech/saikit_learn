import numpy as np
import pytest
from saikit_learn.decision_tree import DecisionTreeRegressor


@pytest.fixture
def regression_dataset():
  """Dataset 1D simples de regressão linear para validações rápidas."""
  X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
  y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
  return X, y


@pytest.fixture
def non_linear_dataset():
  """Dataset 2D não-linear para validar múltiplos splits."""
  X = np.array([
      [0.0, 1.0],
      [1.0, 1.0],
      [5.0, 5.0],
      [6.0, 5.0],
      [10.0, 10.0],
      [11.0, 10.0],
  ])
  y = np.array([10.0, 10.0, 50.0, 50.0, 100.0, 100.0])
  return X, y


# --- 1. TESTES DE INICIALIZAÇÃO E PARÂMETROS ---


# def test_init_invalid_parameters():
#   """Garante que a classe levanta ValueError para critérios ou estratégias inválidas."""
#   with pytest.raises(ValueError):
#     DecisionTreeRegressor(criterion="absolute_error")

#   with pytest.raises(ValueError):
#     DecisionTreeRegressor(split_strategy="random")


# # --- 2. TESTES DE ESTADO NÃO TREINADO ---


# def test_check_fitted_exceptions(regression_dataset):
#   """Testa se chamar métodos antes do fit levanta RuntimeError."""
#   X, _ = regression_dataset
#   reg = DecisionTreeRegressor()

#   with pytest.raises(RuntimeError):
#     _ = reg.depth

#   with pytest.raises(RuntimeError):
#     _ = reg.n_leaves

#   with pytest.raises(RuntimeError):
#     reg.predict(X)

# def test_impurity_calculation():
#   """Verifica a exatidão matemática do cálculo de impureza (variância)."""
#   y_pure = np.array([5.0, 5.0, 5.0])
#   assert DecisionTreeRegressor._impurity(y_pure) == pytest.approx(0.0)

#   y_var = np.array([2.0, 4.0, 6.0])
#   expected_variance = float(np.var(y_var))
#   assert DecisionTreeRegressor._impurity(y_var) == pytest.approx(
#       expected_variance
#   )

# def test_tree_depth_and_max_depth(regression_dataset):
#   """Testa se a árvore respeita o limite de max_depth estipulado."""
#   X, y = regression_dataset
#   reg = DecisionTreeRegressor(max_depth=1)
#   reg.fit(X, y)

#   assert reg.depth <= 1


# def test_min_samples_split(regression_dataset):
#   """Verifica se min_samples_split alto impede novas ramificações."""
#   X, y = regression_dataset
#   reg = DecisionTreeRegressor(min_samples_split=10)
#   reg.fit(X, y)

#   assert reg.depth == 0
#   assert reg.n_leaves == 1


# # --- 5. TESTES DE TREINAMENTO, PREDIÇÃO E R² SCORE ---


# def test_tree_fit_and_properties(regression_dataset):
#   """Testa se o método fit ajusta os atributos internos da árvore."""
#   X, y = regression_dataset
#   reg = DecisionTreeRegressor()
#   reg.fit(X, y)

#   assert reg.tree is not None
#   assert reg.n_features_ == 1
#   assert reg.n_leaves > 0


def test_predict_and_r2_score(regression_dataset):
  """Testa a precisão das predições contínuas e o cálculo do R² score."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(max_depth=3)
  reg.fit(X, y)

  predictions = reg.predict(X)
  assert predictions.shape == y.shape
  assert isinstance(predictions[0], (float, np.floating))

  r2 = reg.score(X, y)
  assert r2 == pytest.approx(1.0, abs=0.1)


def test_predict_single_sample(regression_dataset):
  """Garante que a predição para um único vetor 1D funciona corretamente."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor()
  reg.fit(X, y)

  single_x = np.array([3.0])
  pred = reg.predict(single_x)
  assert pred.shape == (1,)
  assert pred[0] == pytest.approx(6.0)

def test_split_strategy_median(regression_dataset):
  """Verifica se a estratégia de split por mediana executa sem erros."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(split_strategy="median")
  reg.fit(X, y)

  predictions = reg.predict(X)
  assert len(predictions) == len(y)
  assert reg.score(X, y) >= 0.0