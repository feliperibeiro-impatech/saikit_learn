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

def R2(X: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calcula o coeficiente de determinação (R²) entre os valores reais e os valores previstos.

    Parametros:
    X (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.

    Retorno:
    float: O coeficiente de determinação (R²).
    """
    X, y_pred = validate_regression_inputs(X, y_pred)
    rss = RSS(X, y_pred)
    tss = np.sum((X - np.mean(X)) ** 2)
    if tss == 0: #validação especifica de R2, para não dividir por zero 
        raise ValueError("A variância dos valores reais é zero, não é possível calcular R²")
    return float(1 - (rss / tss))

def R2_ajusted(X: NDArray[np.float64], y_pred: NDArray[np.float64], p: int) -> float:
    """
    Calcula o coeficiente de determinação ajustado (R² ajustado) entre os valores reais e os valores previstos.

    Parametros:
    X (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.
    p (int): O número de variáveis independentes no modelo, sem o intercepto.

    Retorno:
    float: O coeficiente de determinação ajustado (R² ajustado).

    Raises
    ------
    TypeError
        Se p não for um número inteiro ou se for um booleano.
    ValueError
        Se p for negativo ou se o número de observações for menor ou igual a p + 1.
    """
    X, y_pred = validate_regression_inputs(X, y_pred)
    if not isinstance(p, (int, np.integer)) or isinstance(p, (bool, np.bool_)):
        raise TypeError("p deve ser um número inteiro")
    n = X.size
    r2 = R2(X, y_pred)
    if p < 0:
        raise ValueError("O número de variáveis independentes deve ser positivo")
    if n <= p + 1:
        raise ValueError("O número de observações deve ser maior que p + 1")
    return float(1-(1-r2)*((n-1)/(n-p-1)))

def RSE(X: NDArray[np.float64], y_pred: NDArray[np.float64], p: int) -> float:
    """
    Calcula o erro padrão residual (RSE) entre os valores reais e os valores previstos.

    Parametros:
    X (NDArray[np.float64]): Os valores reais.
    y_pred (NDArray[np.float64]): Os valores previstos.
    p (int): O número de variáveis independentes no modelo, sem o intercepto.

    Retorno:
    float: O erro padrão residual (RSE).

    Raises
    ------
    TypeError
        Se p não for um número inteiro ou se for um booleano.
    ValueError
        Se p for negativo ou se o número de observações for menor ou igual a p +1
    """
    X, y_pred = validate_regression_inputs(X, y_pred)
    n = X.size
    if not isinstance(p, (int, np.integer)) or isinstance(p, (bool, np.bool_)):
        raise TypeError("p deve ser um número inteiro")
    if p < 0:
        raise ValueError("O número de variáveis independentes deve ser positivo")
    if n <= p + 1:
        raise ValueError("O número de observações deve ser maior que p + 1")

    rss = RSS(X, y_pred)
    return float(np.sqrt(rss / (n - p - 1)))