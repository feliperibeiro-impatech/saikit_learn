from saikit_learn.regression import Ridge
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
def test_recovers_coeficients_without_noise_OLS(data):
    X, y = data
    model = Ridge(lmbd=0).fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, interception, atol=1e-8)


# Teste de Erros
@pytest.mark.parametrize(
    'lmbd',
    [
        np.nan,
        np.inf,
        'str',
        [],
        {1 : 2},
        set(),
        -1,
        True
    ]
)

def test_invalid_lambda(data, lmbd):
    X, y = data 
    model = Ridge().fit(X, y)
    with pytest.raises(ValueError):
        Ridge(lmbd).fit(X, y)
