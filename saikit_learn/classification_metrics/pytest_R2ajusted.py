import numpy as np
import pytest
from regression import R2_ajusted

def test_R2_ajusted_1_perfect_prediction():
    """
    Testa o R2 ajustado com previsões perfeitas.
    O R2 ajustado deve retornar 1.0.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert R2_ajusted(X, y_pred, p=1) == pytest.approx(1.0)

def test_R2_ajusted_2_simple_case():
    """
    Testa o R2 ajustado com um caso simples.
    O R2 ajustado deve retornar 0.898.
    """
    X = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 35.0, 39.0])
    assert R2_ajusted(X, y_pred, p=1) == pytest.approx(0.898)

def test_R2_ajusted_3_negative_values():
    """
    Testa o R2 ajustado com valores negativos.
    O R2 ajustado deve retornar -4.0.
    """
    X = np.array([-2.0, -1.0, 0.0])
    y_pred = np.array([-1.0, 1.0, 0.0])
    assert R2_ajusted(X, y_pred, p=1) == pytest.approx(-4.0)

def test_R2_ajusted_4_large_numbers():
    """
    Testa o R2 ajustado com números grandes.
    O R2 ajustado deve retornar aproximadamente 1.0.
    """
    X = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert R2_ajusted(X, y_pred, p=1) == pytest.approx(1.0)

def test_R2_ajusted_5_small_numbers():
    """
    Testa o R2 ajustado com números pequenos.
    O R2 ajustado deve retornar 0.97.
    """
    X = np.array([1e-10, 2e-10, 3e-10])
    y_pred = X + 1e-11
    assert R2_ajusted(X, y_pred, p=1) == pytest.approx(0.97)

def test_R2_ajusted_6_return_type():
    """
    Testa o tipo de retorno do R2 ajustado.
    O R2 ajustado deve retornar um float.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    assert isinstance(R2_ajusted(X, y_pred, p=1), float)

def test_R2_ajusted_7_zero_score():
    """
    Testa o R2 ajustado quando o R2 tradicional é zero.
    Com n=3 e p=1, o R2 ajustado deve retornar -1.0.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert R2_ajusted(X, y_pred, p=1) == pytest.approx(-1.0)

def test_R2_ajusted_8_constant_true_values():
    """
    Testa o R2 ajustado com valores reais constantes.
    Como TSS = 0, deve lançar ValueError.
    """
    X = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        R2_ajusted(X, y_pred, p=1)

def test_R2_ajusted_9_invalid_degrees_of_freedom():
    """
    Testa o R2 ajustado com graus de liberdade inválidos.
    Deve lançar ValueError quando n <= p + 1.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        R2_ajusted(X, y_pred, p=2)

def test_R2_ajusted_10_negative_predictors():
    """
    Testa o R2 ajustado com número negativo de preditores.
    Deve lançar ValueError.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        R2_ajusted(X, y_pred, p=-1)

def test_R2_ajusted_11_non_integer_predictors():
    """
    Testa o R2 ajustado com número não inteiro de preditores.
    Deve lançar TypeError.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    with pytest.raises(TypeError):
        R2_ajusted(X, y_pred, p=1.5)

def test_R2_ajusted_12_boolean_predictors():
    """
    Testa o R2 ajustado com p booleano.
    Deve lançar TypeError.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    with pytest.raises(TypeError):
        R2_ajusted(X, y_pred, p=True)