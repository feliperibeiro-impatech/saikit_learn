import numpy as np
import typing as tp
import numpy.typing as npt
from numpy.typing import NDArray

""" ANOTAÇÕES E DUVIDAS -- EM ANDAMENTO - NÃO ESQUECER DE REMOVER AO FINAL
Não esquecer da DocString!!! Incluir ao final
Pessoal está implementando em ingles ou portugues?
Docstring em portugues e funções em ingles e mensagens de erros em portugues  

PRECISO GARANTIR ALGUMAS VALIDAÇÕES DE ENTRADA, COMO MESMO TAMANHO, NÃO VAZIO E ETC!!! ACHO QUE ESSAS
VALIDAÇÕES DEVEM SE REPETIR PARA AS DEMAIS MÉTRICAS... TALVEZ SEJA INTERESSANTE SEGMENTAR ESSES TESTES 

Avaliar com grupo se coloca o nome completo das metricas ou se deixa abreviado assim 

dados reais (y_true = x (estão sendo chamados assim pelo grupo))

"""


def MSE(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calcula o erro quadrático médio (MSE) entre os valores reais e os valores previstos.

    Parametros:
    y_true (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.

    Retorno:
    float: O erro quadrático médio.
    """

    squared_errors = (y_true - y_pred) ** 2
    return float(np.mean(squared_errors))

def RMSE(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calcula o erro quadrático médio da raiz (RMSE) entre os valores reais e os valores previstos.

    Parametros:
    y_true (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.

    Retorno:
    float: O erro quadrático médio da raiz.
    """
    return np.sqrt(MSE(y_true, y_pred))

def MAE(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calcula o erro absoluto médio (MAE) entre os valores reais e os valores previstos.

    Parametros:
    y_true (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.

    Retorno:
    float: O erro absoluto médio.
    """
    absolute_errors = np.abs(y_true - y_pred)
    return float(np.mean(absolute_errors))

def RSS(): #DAVI
    pass

def R2(): #DAVI
    pass

def R2_ajustado(): #DAVI
    pass

def RSE(): #DAVI
    pass

