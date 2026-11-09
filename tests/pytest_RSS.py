
import numpy as np
import pytest
from regression import RSS

def test_RSS_1_perfect_prediction():
    """
    Testa a função RSS com previsões perfeitas.
    O RSS deve retornar 0 quando os valores previstos são exatamente iguais aos valores reais.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert RSS(X, y_pred) == pytest.approx(0.0)

def test_RSS_2_simple_case():
    """"
    Testa a função RSS com um caso simples.
    O RSS deve retornar 34
    """
    X = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 35.0, 39.0])
    assert RSS(X, y_pred) == pytest.approx(34)

def test_RSS_3_negative_values():
    """"
    Testa a função RSS com valores negativos.
    O RSS deve retornar 5
    """
    X = np.array([-2.0, -1.0, 0.0])
    y_pred = np.array([-1.0, 1.0, 0.0])
    assert RSS(X, y_pred) == pytest.approx(5)

def test_RSS_4_large_numbers():
    """"
    Testa a função RSS com números grandes.
    O RSS deve retornar 19200
    """
    X = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert RSS(X, y_pred) == pytest.approx(19200.0)

def test_RSS_5_small_numbers():
    """"
    Testa a função RSS com números pequenos.
    O RSS deve retornar 3e-22
    """
    X = np.array([1e-10, 2e-10, 3e-10])
    y_pred = X + 1e-11
    assert RSS(X, y_pred) == pytest.approx(3e-22,rel=1e-6,abs=0)

def test_RSS_6_symmetry():
    """"
    Testa a propriedade de simetria da função RSS.
    O RSS deve ser o mesmo quando os valores reais e previstos são trocados."""
    X = np.array([1.0, 4.0, 7.0])
    y_pred = np.array([2.0, 3.0, 9.0])
    assert RSS(X, y_pred) == pytest.approx(RSS(y_pred, X))

def test_RSS_7_return_type():
    """"
    Testa o tipo de retorno da função RSS.
    O RSS deve retornar um valor do tipo float.
    """
    X = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 2.0])
    assert isinstance(RSS(X, y_pred), float)
