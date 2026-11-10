import numpy as np
import pytest
from saikit_learn.regression_metrics.regression import R2


def test_R2_1_perfect_prediction():
    """
    Testa a função R2 com previsões perfeitas.
    O R2 deve retornar 1.0 quando os valores previstos são exatamente iguais aos valores reais.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert R2(X, y_pred) == pytest.approx(1.0)

def test_R2_2_simple_case():
    """"
    Testa a função R2 com um caso simples.
    O R2 deve retornar 0.5
    """
    X = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 35.0, 39.0])
    assert R2(X, y_pred) == pytest.approx(0.932)

def test_R2_3_negative_values():
    """"
    Testa a função R2 com valores negativos.
    O R2 deve retornar -1.5
    """
    X = np.array([-2.0, -1.0, 0.0])
    y_pred = np.array([-1.0, 1.0, 0.0])
    assert R2(X, y_pred) == pytest.approx(-1.5)

def test_R2_4_large_numbers():
    """"
    Testa a função R2 com números grandes.
    O R2 deve retornar 80
    """
    X = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert R2(X, y_pred) == pytest.approx(1.0)

def test_R2_5_small_numbers(): 
    """"
    Testa a função R2 com números pequenos.
    O R2 deve retornar 1e-11
    """
    X = np.array([1e-10, 2e-10, 3e-10])
    y_pred = X + 1e-11
    assert R2(X, y_pred) == pytest.approx(0.985)

def test_R2_6_return_type():
    """"
    Testa o tipo de retorno da função R2.
    O R2 deve retornar um valor do tipo float.
    """
    X = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 2.0])
    assert isinstance(R2(X, y_pred), float)

def test_R2_7_zero_score():
    """
    Prever sempre a média dos valores reais produz R2 = 0.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert R2(X, y_pred) == pytest.approx(0.0)

def test_R2_8_constant_true_values():
    """
    Se TSS = 0, o R2 é indefinido.
    """
    X = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    with pytest.raises(ValueError): 
        R2(X, y_pred)
