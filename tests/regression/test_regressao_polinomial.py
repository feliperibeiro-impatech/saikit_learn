from saikit_learn.regression import PolReg, OLS
from saikit_learn.regression.utils.validacao import NotFittedError
import numpy as np
import pytest

# Setup Inicial
def pol(x):
    return 1.2 + 3.4 * x - 7.34 * x ** 2 + 0.3 * x ** 3

@pytest.fixture
def data():
    '''X em [-10, 10] (100, 1) e y exatamente cúbico, sem ruído'''
    rng = np.random.default_rng(0)
    X = rng.uniform(-10, 10, size = (100, 1))
    return X, pol(X[:, 0])

@pytest.fixture
def noise_data():
    '''X em [-10, 10] (300, 1) e y cúbico com ruído'''
    rng = np.random.default_rng(1)
    X = rng.uniform(-10, 10, size = (300, 1))
    y = pol(X[:, 0]) + rng.normal(scale = 0.5, size = 300)
    return X, y

# Testes Numéricos
def test_recovers_exact_poly(data):
    X, y = data
    model = PolReg(d=3).fit(X, y)
    X_new = np.linspace(-10, 10, 100)[:, None]
    np.testing.assert_allclose(model.predict(X_new), pol(X_new[:, 0]), atol = 1e-8)

def test_recovers_exact_poly_coef(data):
    X, y = data
    model = PolReg(d=3).fit(X, y)
    np.testing.assert_allclose(model.model_.coef_, [3.4, -7.34, 0.3], atol = 1e-8)
    np.testing.assert_allclose(model.model_.intercept_, 1.2, atol = 1e-8)

def test_recovers_noise_poly_coef(noise_data):
    X, y = noise_data
    model = PolReg(d=3).fit(X, y)
    X_new = np.linspace(-10, 10, 100)[:, None]
    np.testing.assert_allclose(model.predict(X_new), pol(X_new[:, 0]), atol = 0.5)

def text_recover_exact_poly_without_coef(data):
    X, y = data
    model = PolReg(d=3, intercepto=False).fit(X, y)
    X_new = np.linspace(-10, 10, 100)[:, None]
    np.testing.assert_allclose(model.predict(X_new) + 1.2, pol(X_new[:, 0]), atol = 1e-8)

def test_degree_one_equals_ols(noise_data):
    X, y = noise_data
    model_pol = PolReg(d=1).fit(X, y)
    model_ols = OLS().fit(X, y)
    np.testing.assert_allclose(model_pol.predict(X), model_ols.predict(X), atol = 1e-8)


@pytest.mark.parametrize(
    'd',
    [
        1,
        3,
        5,
        10,
        15
    ]
)
def test_design_matrix_is_correct(noise_data, d):
    X, y = noise_data
    design = np.hstack([X ** k for k in range(1, d+1)])
    model_ols = OLS().fit(design, y)
    model_pol = PolReg(d).fit(X, y)
    np.testing.assert_allclose(model_pol.predict(X), model_ols.predict(design), atol = 1e-8)


