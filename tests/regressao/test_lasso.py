from saikit_learn.regressao.lasso import Lasso
from saikit_learn.utilidades.validacao import NotFittedError
import numpy as np
import pytest

coef = np.array([1.0, -2.0, 0.3])
interception = 4.5

@pytest.fixture
def data():
    '''X (50, 3) aleatório e y (50,) exato, sem ruído'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ coef + interception
    return X, y


# Testes Numéricos
def test_recovers_coeficients_without_noise_OLS(data):
    X, y = data
    model = Lasso(lmbd=0).fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, atol=1e-6)
    np.testing.assert_allclose(model.intercept_, interception, atol=1e-6)

def test_soft_thresholding_by_hand():
    # (1/n)||x||^2 = 1 e rho = 2, então w = S(2, 0.5) = 1.5
    x = np.array([1.0, -1.0] * 5).reshape(-1, 1)
    model = Lasso(lmbd=0.5).fit(x, 2 * x[:, 0])
    np.testing.assert_allclose(model.coef_, [1.5], atol=1e-8)

def test_large_lambda_zeroes_everything(data):
    X, y = data
    lmax = np.max(np.abs((X - X.mean(0)).T @ (y - y.mean()))) / len(y)
    model = Lasso(lmbd=lmax * 1.01).fit(X, y)
    assert np.all(model.coef_ == 0)
    np.testing.assert_allclose(model.intercept_, y.mean())

def test_predict_values(data):
    X, y = data
    pred = Lasso(lmbd=0).fit(X, y).predict(X)
    assert pred.shape == (50,)
    np.testing.assert_allclose(pred, y, atol=1e-5)


# Teste de Erros
def test_predict_before_fit():
    with pytest.raises(NotFittedError):
        Lasso().predict([[1.0, 2.0]])

def test_predict_wrong_number_of_features(data):
    X, y = data
    model = Lasso().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.ones((4, 2)))

def test_incompatible_sizes(data):
    X, y = data
    with pytest.raises(ValueError):
        Lasso().fit(X, y[:-1])

@pytest.mark.parametrize('lmbd', [np.nan, np.inf, 'str', [], -1, True])
def test_invalid_lambda(lmbd):
    with pytest.raises(ValueError):
        Lasso(lmbd)