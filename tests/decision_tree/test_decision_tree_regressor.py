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


@pytest.fixture
def step_dataset():
  """Degrau 1D: y = 1 para x <= 3 e y = 9 para x >= 4 (corte ótimo em 3.5)."""
  X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
  y = np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
  return X, y


# --- 1. TESTES DE INICIALIZAÇÃO E PARÂMETROS ---


def test_init_invalid_parameters():
  """Garante que a classe levanta ValueError para critérios ou estratégias inválidas."""
  with pytest.raises(ValueError):
    DecisionTreeRegressor(criterion="absolute_error")

  with pytest.raises(ValueError):
    DecisionTreeRegressor(split_strategy="random")


def test_init_default_state():
  """Antes do fit não existe árvore nem número de features."""
  reg = DecisionTreeRegressor()

  assert reg.tree is None
  assert reg.n_features is None


# --- 2. TESTES DE ESTADO NÃO TREINADO ---


def test_check_fitted_exceptions(regression_dataset):
  """Testa se chamar métodos antes do fit levanta ValueError."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor()

  with pytest.raises(ValueError):
    _ = reg.depth

  with pytest.raises(ValueError):
    _ = reg.n_leaves

  with pytest.raises(ValueError):
    reg.predict(X)

  with pytest.raises(ValueError):
    reg.score(X, y)


# --- 3. TESTES DE IMPUREZA ---


def test_impurity_calculation():
  """Verifica a exatidão matemática do cálculo de impureza (variância)."""
  y_pure = np.array([5.0, 5.0, 5.0])
  assert DecisionTreeRegressor._impurity(y_pure) == pytest.approx(0.0)

  y_var = np.array([2.0, 4.0, 6.0])
  expected_variance = float(np.var(y_var))
  assert DecisionTreeRegressor._impurity(y_var) == pytest.approx(
      expected_variance
  )

  # Resíduos simétricos: a média dos resíduos é 0, mas a variância não.
  assert DecisionTreeRegressor._impurity(np.array([0.0, 2.0])) == pytest.approx(
      1.0
  )


# --- 4. TESTES DE CORTE ---


def test_best_split_on_step(step_dataset):
  """O melhor corte de um degrau fica entre o 3 e o 4, com erro zero."""
  X, y = step_dataset
  reg = DecisionTreeRegressor()

  feature, threshold, score = reg._get_best_split(X, y)

  assert feature == 0
  assert threshold == pytest.approx(3.5)
  assert score == pytest.approx(0.0)


def test_best_split_picks_informative_feature():
  """Entre duas features, escolhe a que separa os valores de y."""
  X = np.array([[7.0, 1.0], [3.0, 2.0], [9.0, 8.0], [1.0, 9.0]])
  y = np.array([0.0, 0.0, 10.0, 10.0])
  reg = DecisionTreeRegressor()

  feature, threshold, _ = reg._get_best_split(X, y)

  assert feature == 1
  assert threshold == pytest.approx(5.0)


def test_best_split_none_when_feature_is_constant():
  """Feature constante não tem corte possível."""
  X = np.ones((4, 1))
  y = np.array([1.0, 2.0, 3.0, 4.0])

  assert DecisionTreeRegressor()._get_best_split(X, y) is None


def test_candidate_thresholds_all():
  """Estratégia 'all' usa o ponto médio entre valores distintos consecutivos."""
  reg = DecisionTreeRegressor(split_strategy="all")

  thresholds = reg._get_candidate_thresholds(np.array([1.0, 2.0, 2.0, 4.0]))

  assert thresholds == pytest.approx(np.array([1.5, 3.0]))


# --- 5. TESTES DE TREINAMENTO ---


def test_tree_fit_and_properties(regression_dataset):
  """Testa se o método fit ajusta os atributos internos da árvore."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor()
  result = reg.fit(X, y)

  assert result is reg
  assert reg.tree is not None
  assert reg.n_features == 1
  assert reg.n_leaves > 0


def test_train_is_alias_of_fit(regression_dataset):
  """train deve devolver o próprio modelo treinado."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor()

  assert reg.train(X, y) is reg
  assert reg.tree is not None


def test_fit_invalid_shapes(regression_dataset):
  """X precisa ser 2D, y precisa ser 1D e os dois com o mesmo número de linhas."""
  X, y = regression_dataset

  with pytest.raises(ValueError):
    DecisionTreeRegressor().fit(X.ravel(), y)

  with pytest.raises(ValueError):
    DecisionTreeRegressor().fit(X, y.reshape(-1, 1))

  with pytest.raises(ValueError):
    DecisionTreeRegressor().fit(X, y[:-1])


def test_fit_step_function(step_dataset):
  """Um degrau é ajustado com um único corte, no ponto médio."""
  X, y = step_dataset
  reg = DecisionTreeRegressor().fit(X, y)

  assert reg.depth == 1
  assert reg.n_leaves == 2
  assert reg.tree.threshold == pytest.approx(3.5)
  assert reg.predict(X) == pytest.approx(y)


def test_leaf_value_is_mean():
  """A previsão de cada folha é a média dos y que caíram nela."""
  X = np.array([[1.0], [2.0], [3.0], [4.0]])
  y = np.array([0.0, 2.0, 10.0, 14.0])
  reg = DecisionTreeRegressor(max_depth=1).fit(X, y)

  assert reg.predict(X) == pytest.approx(np.array([1.0, 1.0, 12.0, 12.0]))


def test_fit_non_linear_dataset(non_linear_dataset):
  """Três patamares de y precisam de 3 folhas e profundidade 2."""
  X, y = non_linear_dataset
  reg = DecisionTreeRegressor().fit(X, y)

  assert reg.n_features == 2
  assert reg.n_leaves == 3
  assert reg.depth == 2
  assert reg.predict(X) == pytest.approx(y)


def test_fit_accepts_python_lists():
  """Listas Python são convertidas para arrays."""
  reg = DecisionTreeRegressor().fit([[1], [2], [8], [9]], [0, 0, 10, 10])

  assert reg.predict([[1.5], [8.5]]) == pytest.approx(np.array([0.0, 10.0]))


def test_constant_target_is_single_leaf():
  """Se todos os y são iguais, não há o que cortar."""
  X = np.array([[1.0], [2.0], [3.0], [4.0]])
  y = np.full(4, 5.0)
  reg = DecisionTreeRegressor().fit(X, y)

  assert reg.n_leaves == 1
  assert reg.predict(X) == pytest.approx(y)


# --- 6. TESTES DOS HIPERPARÂMETROS DE PARADA ---


def test_tree_depth_and_max_depth(regression_dataset):
  """Testa se a árvore respeita o limite de max_depth estipulado."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(max_depth=1)
  reg.fit(X, y)

  assert reg.depth <= 1


def test_max_depth_zero_predicts_mean(regression_dataset):
  """Com max_depth=0 a árvore é só a raiz: prevê a média e o R² é 0."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(max_depth=0).fit(X, y)

  assert reg.n_leaves == 1
  assert reg.predict(X) == pytest.approx(np.full(5, y.mean()))
  assert reg.score(X, y) == pytest.approx(0.0)


def test_min_samples_split(regression_dataset):
  """Verifica se min_samples_split alto impede novas ramificações."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(min_samples_split=10)
  reg.fit(X, y)

  assert reg.depth == 0
  assert reg.n_leaves == 1


def test_min_samples_leaf(regression_dataset):
  """Com min_samples_leaf=2 só dá para cortar uma vez (5 pontos)."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(min_samples_leaf=2).fit(X, y)

  assert reg.depth == 1
  assert reg.n_leaves == 2
  assert reg.tree.left.n_samples >= 2
  assert reg.tree.right.n_samples >= 2


def test_min_samples_leaf_too_large(regression_dataset):
  """Se nenhum corte respeita min_samples_leaf, sobra só a raiz."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(min_samples_leaf=4).fit(X, y)

  assert reg.n_leaves == 1


def test_min_gain(regression_dataset):
  """Um ganho mínimo impossível de atingir impede qualquer corte."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(min_gain=1e6).fit(X, y)

  assert reg.n_leaves == 1


# --- 7. TESTES DE PREDIÇÃO E R² SCORE ---


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


def test_predict_threshold_rule_and_extrapolation(step_dataset):
  """x <= corte vai para a esquerda; fora do intervalo usa a folha mais próxima."""
  X, y = step_dataset
  reg = DecisionTreeRegressor().fit(X, y)

  new_X = np.array([[-100.0], [3.4], [3.5], [3.6], [100.0]])
  assert reg.predict(new_X) == pytest.approx(
      np.array([1.0, 1.0, 1.0, 9.0, 9.0])
  )


def test_score_matches_r2_formula(regression_dataset):
  """R² = 1 - RSS / TSS, conferido na mão para uma árvore de 1 corte."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(max_depth=1).fit(X, y)

  pred = reg.predict(X)
  rss = np.sum((y - pred) ** 2)
  tss = np.sum((y - y.mean()) ** 2)

  assert reg.score(X, y) == pytest.approx(1 - rss / tss)
  assert 0.0 < reg.score(X, y) < 1.0


def test_score_with_constant_y_is_zero(regression_dataset):
  """Se TSS = 0 o score devolve 0 em vez de dividir por zero."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor().fit(X, y)

  assert reg.score(X, np.full(5, 3.0)) == 0.0


# --- 8. TESTES DA ESTRATÉGIA DE CORTE ---


def test_split_strategy_median(regression_dataset):
  """Verifica se a estratégia de split por mediana executa sem erros."""
  X, y = regression_dataset
  reg = DecisionTreeRegressor(split_strategy="median")
  reg.fit(X, y)

  predictions = reg.predict(X)
  assert len(predictions) == len(y)
  assert reg.score(X, y) >= 0.0


def test_split_strategy_median_cuts_at_median(step_dataset, regression_dataset):
  """O primeiro corte da estratégia 'median' é a mediana da feature."""
  X_even, y_even = step_dataset
  reg = DecisionTreeRegressor(split_strategy="median", max_depth=1)
  reg.fit(X_even, y_even)
  assert reg.tree.threshold == pytest.approx(3.5)  # mediana de 1..6

  X_odd, y_odd = regression_dataset
  reg = DecisionTreeRegressor(split_strategy="median", max_depth=1)
  reg.fit(X_odd, y_odd)
  assert reg.tree.threshold == pytest.approx(3.0)  # mediana de 1..6