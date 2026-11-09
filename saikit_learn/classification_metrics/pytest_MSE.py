import numpy as np
import pytest 

from regression import MSE

def test_MSE_0_perfect_prediction():
    """
    Teste da função MSE com previsões perfeitas.
    O MSE deve ser 0 quando os valores previstos são exatamente iguais aos valores reais.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert MSE(X, y_pred) == 0.0

def test_MSE_1_sp_case():
    """
    Teste da função MSE com um caso simples.
    O MSE deve ser 1
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert MSE(X, y_pred) == 1.0

def test_MSE_2_sp_case():
    """
    Teste da função MSE com um caso simples.
    O MSE deve ser 2/3
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 2.0])
    assert MSE(X, y_pred) == pytest.approx(2/3)

def test_MSE_3_negative_case():
    """
    Teste da função MSE com um caso simples negativo.
    O MSE deve ser 5/3
    """
    X = np.array([-2.0, -1.0, 0.0])
    y_pred = np.array([-1.0, 1.0, 0.0])
    assert MSE(X, y_pred) == pytest.approx(5/3)

def test_MSE_4_decimal_case():
    """
    Teste da função MSE com um caso decimal simples.
    O MSE deve ser 1/4
    """
    X = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 1.5])
    assert MSE(X, y_pred) == pytest.approx(1/4)

def test_MSE_5_large_numbers_case():
    """
    Teste da função MSE com números grandes.
    O MSE deve ser 6400.0, que é o quadrado da diferença (80) entre os valores reais e previstos.
    """
    X = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert MSE(X, y_pred) == pytest.approx(6400.0) 

def test_MSE_6_simetric_case():
    """
    Teste da função MSE com um caso simétrico.
    O MSE deve ser o mesmo quando os valores reais e previstos são trocados.
    """
    X = np.array([-1.0, 0.0, 1.0])
    y_pred = np.array([1.0, 0.0, -1.0])
    assert MSE(X, y_pred) == MSE(y_pred, X)

def test_MSE_7_MSE_return():
    """
    Testa o retorno da função MSE.
    O valor de retorno do MSE deve ser um float.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert isinstance(MSE(X, y_pred), float)

def test_MSE_8_constant_case():
    """
    Testa a função MSE com uma previsão constante.
    O MSE deve ser igual ao resultado original multiplicado pelo quadrado da constante.
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 5.0, 2.0])
    c = 4.0
    assert MSE((X * c),(y_pred * c)) == MSE(X, y_pred) * c * c

def test_MSE_9_empty_case():
    """
    Testa a função MSE com arrays vazios.
    O MSE deve retornar um ValueError quando os arrays de entrada estiverem vazios.
    """
    X = np.array([])
    y_pred = np.array([])
    with pytest.raises(ValueError):
        MSE(X, y_pred)

def test_MSE_10_different_length_case():
    """
    Testa a função MSE com arrays de comprimentos diferentes.
    O MSE deve retornar um ValueError quando os arrays de entrada tiverem comprimentos diferentes
    """
    X = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        MSE(X, y_pred)

def test_MSE_11_non_numeric_case():
    """
    Testa a função MSE com entrada não numérica.
    O MSE deve retornar um TypeError quando os arrays de entrada contiverem valores não
    """
    X = np.array([1.0, 2.0, 'a'])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        MSE(X, y_pred)

def test_MSE_12_nan_case():
    """
    Testa a função MSE com valores NaN.
    O MSE deve retornar um ValueError quando os arrays de entrada contiverem valores Na
    """
    X = np.array([1.0, 2.0, np.nan])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        MSE(X, y_pred)

def test_MSE_13_inf_case():
    """
    Testa a função MSE com valores infinitos.
    O MSE deve retornar um ValueError quando os arrays de entrada contiverem valores infinit
    """
    X = np.array([1.0, 2.0, np.inf])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        MSE(X, y_pred)

