import numpy as np
import typing as tp
import numpy.typing as npt
from numpy.typing import NDArray
from _validation import validate_regression_inputs

""" ANOTAÇÕES E DUVIDAS -- EM ANDAMENTO - NÃO ESQUECER DE REMOVER AO FINAL
CONVENÇÃO GRUPO: x = y_true e y= y_pred 
VERIFICAR CONVENÇÃO DE Y_PRED SE É SOMENTE Y MESMO.
Docstring em portugues, assim como mensagens de erro 
Funções e variáveis em ingles. 

Avaliar se deixo nome das funções abreviado ou completo. Ver como andre alinhou o dele 

PRECISO GARANTIR ALGUMAS VALIDAÇÕES DE ENTRADA, COMO MESMO TAMANHO, NÃO VAZIO E ETC!!! ACHO QUE ESSAS
VALIDAÇÕES DEVEM SE REPETIR PARA AS DEMAIS MÉTRICAS... TALVEZ SEJA INTERESSANTE SEGMENTAR ESSES TESTES 

Tem como parametrizar os testes de validação, sem precisar ficar repetindo o mesmo teste para cada métrica.... !!! Bom ajuste!

"""




def MSE(X: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calcula o erro quadrático médio (MSE) entre os valores reais e os valores previstos.

    Parametros:
    X (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.

    Retorno:
    float: O erro quadrático médio.
    """
    X, y_pred = validate_regression_inputs(X, y_pred)
    squared_errors = (X - y_pred) ** 2
    return float(np.mean(squared_errors))

def RMSE(X: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calcula o erro quadrático médio da raiz (RMSE) entre os valores reais e os valores previstos.

    Parametros:
    X (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.

    Retorno:
    float: O erro quadrático médio da raiz.
    """
    X, y_pred = validate_regression_inputs(X, y_pred)
    return np.sqrt(MSE(X, y_pred))

def MAE(X: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calcula o erro absoluto médio (MAE) entre os valores reais e os valores previstos.

    Parametros:
    X (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.

    Retorno:
    float: O erro absoluto médio.
    """
    X, y_pred = validate_regression_inputs(X, y_pred)
    absolute_errors = np.abs(X - y_pred)
    return float(np.mean(absolute_errors))

def RSS(X: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calcula a soma dos quadrados dos resíduos (RSS) entre os valores reais e os valores previstos.

    Parametros:
    X (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.

    Retorno:
    float: A soma dos quadrados dos resíduos.
    """
    X, y_pred = validate_regression_inputs(X, y_pred)
    squared_errors = (X - y_pred) ** 2
    return float(np.sum(squared_errors))

def R2(): 
    pass

def R2_ajustado(): #DAVI
    pass

def RSE(): #DAVI
    pass

