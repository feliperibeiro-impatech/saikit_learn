import numpy as np
import pytest 

from regression import RMSE

# já que esse método está baseado na função MSE, previamente testada, vamos apenas testar alguns casos para ver se está funcionando adequadamente. 

def test_RMSE_1_large_numbers_case():
    """
    Teste da função MSE com números grandes.
    O MSE deve retornar 80
    """
    y_true = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert RMSE(y_true, y_pred) == pytest.approx(80.0) 

def test_RMSE_2_decimal_case():
    """
    Teste da função MSE com um caso decimal simples.
    O MSE deve retornar 1/4
    """
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 1.5])
    assert RMSE(y_true, y_pred) == pytest.approx(0.5)

def test_RMSE_3_perfect_prediction():
    """
    Testa a função MSE com previsões perfeitas.
    O MSE deve ser 0 quando os valores previstos são exatamente iguais aos valores reais
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert RMSE(y_true, y_pred) == 0.0

def test_RMSE_4_small_numbers_case():
    """
    Testa a função MSE com números pequenos.
    O MSE deve ser 1e-11
    """
    y_true = np.array([1e-10, 2e-10, 3e-10])
    y_pred = y_true + 1e-11  
    assert RMSE(y_true, y_pred) == pytest.approx(1e-11, rel=1e-6, abs=0)
