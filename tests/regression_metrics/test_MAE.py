import numpy as np
import pytest
from saikit_learn.regression_metrics.regression import MAE

def test_MAE_1_perfect_prediction():
    """
    Testa a função MAE com previsões perfeitas.
    O MAE deve retornar 0 quando os valores previstos são exatamente iguais aos valores reais.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert MAE(X, y_pred) == pytest.approx(0.0)

def test_MAE_2_simple_case():
    """"
    Testa a função MAE com um caso simples.
    O MAE deve retornar 2.5
    """
    X = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 35.0, 39.0])
    assert MAE(X, y_pred) == pytest.approx(2.5)

def test_MAE_3_negative_values():
    """"
    Testa a função MAE com valores negativos.
    O MAE deve retornar 1
    """
    X = np.array([-2.0, -1.0, 0.0])
    y_pred = np.array([-1.0, 1.0, 0.0])
    assert MAE(X, y_pred) == pytest.approx(1.0)

def test_MAE_4_large_numbers():
    """"
    Testa a função MAE com números grandes.
    O MAE deve retornar 80
    """
    X = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert MAE(X, y_pred) == pytest.approx(80.0)

def test_MAE_5_small_numbers():
    """"
    Testa a função MAE com números pequenos.
    O MAE deve retornar 1e-11
    """
    X = np.array([1e-10, 2e-10, 3e-10])
    y_pred = X + 1e-11
    assert MAE(X, y_pred) == pytest.approx(1e-11,rel=1e-6,abs=0)

def test_MAE_6_symmetry():
    """"
    Testa a propriedade de simetria da função MAE.
    O MAE deve ser o mesmo quando os valores reais e previstos são trocados."""
    X = np.array([1.0, 4.0, 7.0])
    y_pred = np.array([2.0, 3.0, 9.0])
    assert MAE(X, y_pred) == pytest.approx(MAE(y_pred, X))

def test_MAE_7_return_type():
    """"
    Testa o tipo de retorno da função MAE.
    O MAE deve retornar um valor do tipo float.
    """
    X = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 2.0])
    assert isinstance(MAE(X, y_pred), float)

def test_MAE_8_scale_property():
    """"
    Testa a propriedade de escala da função MAE.
    O MAE deve ser proporcional ao fator de escala aplicado aos valores reais e previstos.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 4.0, 6.0])
    c = -5.0
    assert MAE(c * X, c * y_pred) == pytest.approx(abs(c) * MAE(X, y_pred))