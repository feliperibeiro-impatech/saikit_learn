from saikit_learn.regression.minimos_quadrados import OLS
from saikit_learn.regression.utils.validacao import NotFittedError
import numpy as np
import pytest

# Setup Inicial
coef = np.array([1.0, -2.0, 0.3])
interception = 4.5

@pytest.fixture
def data():
    '''X (50, 3) aleatório e y (50, 1) exato, sem ruído'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ coef + interception
    return X, y

@pytest.fixture
def noise_data():
    '''X (100, 3) aleatório e y (100, 1) com ruído'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X @ coef + interception + rng.normal(scale=0.1, size = 100)
    return X, y

# Testes Numéricos
def test_recovers_coeficients_without_noise(data):
    X, y = data
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, interception, atol=1e-8)

def test_design_matrix(noise_data):
    X, y = noise_data
    A = np.column_stack([np.ones(X.shape[0]), X])
    beta_ref = np.linalg.solve(A.T @ A, A.T @ y)
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.intercept_, beta_ref[0], atol=1e-8)
    np.testing.assert_allclose(model.coef_, beta_ref[1:], atol=1e-8)

def test_orthogonal_residual(noise_data):
    X, y = noise_data
    model = OLS().fit(X, y)
    residual = y - model.predict(X)
    np.testing.assert_allclose(residual.sum(), 0.0, atol=1e-8)
    np.testing.assert_allclose(X.T @ residual, 0.0, atol=1e-8)

def test_correct_coeficients_with_noise(noise_data):
    X, y = noise_data
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.intercept_, interception, atol=.05)
    np.testing.assert_allclose(model.coef_, coef, atol=.05)

def test_predict_correctly(data):
    X, y = data
    model = OLS().fit(X, y)
    rng = np.random.default_rng(10)
    X_new = rng.normal(size=(100, 3))
    y_new = X_new @ coef + interception
    np.testing.assert_allclose(model.predict(X_new), y_new, atol=1e-8)

def test_predict_correctly_with_noise(noise_data):
    X, y = noise_data
    model = OLS().fit(X, y)
    rng = np.random.default_rng(1)
    X_new = rng.normal(size=(1000, 3))
    y_new = X_new @ coef + interception + rng.normal(scale=0.1, size = 1000)
    np.testing.assert_allclose(model.predict(X_new), y_new, atol=.5)

def test_without_interception():
    rng = np.random.default_rng(1)
    X = rng.normal(size = (50, 3))
    y = X @ coef
    model = OLS(intercepto=False).fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, atol=1e-8)

def test_sum_constant_to_y(data):
    X, y = data
    base = OLS().fit(X, y)
    model = OLS().fit(X, y - 4.0)
    np.testing.assert_allclose(model.coef_, base.coef_, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, base.intercept_ - 4.0, atol=1e-8)

def test_more_features_than_sample():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 10))
    y = rng.normal(size=5)
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)

# Teste de Erros
def test_colinear_columns(data):
    X, y = data
    X_rd = np.column_stack([X, X[:, 0]])
    with pytest.raises(ValueError):
        OLS().fit(X_rd, y)

def test_predict_before_fit(data):
    X, y = data
    with pytest.raises(NotFittedError):
        OLS().predict(X)

def test_wrong_dimension_predict(data):
    rng = np.random.default_rng(1)
    X, y = data
    model = OLS().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(rng.normal(size=(10, 5)))


def test_attibutes_do_not_exist_yet():
    model = OLS()
    assert not hasattr(model, 'coef_')
    assert not hasattr(model, 'intercept_')

@pytest.mark.parametrize(
    'X, y',
    [
        (np.ones(15), np.ones(15)),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), np.ones(2)),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), np.ones(2)),
        (np.ones((5, 2)), np.ones(4)),
        (np.ones((3, 2)), np.array([1, np.nan, 1.0])),
        (np.ones((5, 2)), np.ones((5, 2))),
        (np.empty((0, 2)), np.empty(0))
    ]
)

def test_invalid_entries(X, y):
    with pytest.raises(ValueError):
        OLS().fit(X, y)

def test_fit_return_self(data):
    X, y = data
    model  = OLS()
    assert model.fit(X, y) is model

def test_ideal_shape(data):
    X, y = data
    model = OLS().fit(X, y)
    assert model.predict(X).shape == (50,)