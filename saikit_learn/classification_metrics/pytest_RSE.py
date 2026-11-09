import numpy as np
import pytest
from regression import RSE

def test_RSE_1_perfect_prediction():
    """
    Testa o RSE com previsões perfeitas.
    O RSE deve retornar 0.0.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert RSE(X, y_pred, p=1) == pytest.approx(0.0)

def test_RSE_2_simple_case():
    """
    Testa o RSE com um caso simples.
    O RSE deve retornar sqrt(17).
    """
    X = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 35.0, 39.0])
    assert RSE(X, y_pred, p=1) == pytest.approx(np.sqrt(17))

def test_RSE_3_negative_values():
    """
    Testa o RSE com valores negativos.
    O RSE deve retornar sqrt(5).
    """
    X = np.array([-2.0, -1.0, 0.0])
    y_pred = np.array([-1.0, 1.0, 0.0])
    assert RSE(X, y_pred, p=1) == pytest.approx(np.sqrt(5))

def test_RSE_4_large_numbers():
    """
    Testa o RSE com números grandes.
    O RSE deve retornar sqrt(19200).
    """
    X = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert RSE(X, y_pred, p=1) == pytest.approx(np.sqrt(19200))


def test_RSE_5_small_numbers():
    """
    Testa o RSE com números pequenos.
    O RSE deve retornar sqrt(3e-22).
    """
    X = np.array([1e-10, 2e-10, 3e-10])
    y_pred = X + 1e-11
    assert RSE(X, y_pred, p=1) == pytest.approx(np.sqrt(3e-22),rel=1e-6,abs=0)

def test_RSE_6_return_type():
    """
    Testa o tipo de retorno do RSE.
    O RSE deve retornar um float.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    assert isinstance(RSE(X, y_pred, p=1), float)

def test_RSE_7_invalid_degrees_of_freedom():
    """
    Testa graus de liberdade insuficientes.
    Deve lançar ValueError quando n <= p + 1.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        RSE(X, y_pred, p=2)

def test_RSE_8_negative_predictors():
    """
    Testa um número negativo de preditores.
    Deve lançar ValueError.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        RSE(X, y_pred, p=-1)

def test_RSE_9_non_integer_predictors():
    """
    Testa um número não inteiro de preditores.
    Deve lançar TypeError.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    with pytest.raises(TypeError):
        RSE(X, y_pred, p=1.5)

def test_RSE_10_boolean_predictors():
    """
    Testa p booleano.
    Deve lançar TypeError.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    with pytest.raises(TypeError):
        RSE(X, y_pred, p=True)

def test_RSE_11_zero_predictors():
    """
    Testa o RSE com zero preditores.
    O denominador deve ser n - 1.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    assert RSE(X, y_pred, p=0) == pytest.approx(np.sqrt(1 / 2))