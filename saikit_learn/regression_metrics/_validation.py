## arquivo para testes de entradas para validação.
#Validação de entrada de tamanho igual 
#Validação de entrada diferente de vazia 

import numpy as np
from numpy.typing import NDArray

def validate_regression_inputs(X: np.ndarray,y_pred: np.ndarray) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """"
    Validação de entradas para as métricas de regressão.

    Parametros
    ----------
    X : np.ndarray Array contendo valores reais.
    y_pred : np.ndarray Array contendo valores preditos.

    Retorno
    -------
    tuple of NDArray[np.float64] Arrays validados de uma dimensão com valores numéricos.

    Raises
    ------
    TypeError
        Se a entrada não for um array NumPy ou contiver
        tipos de dados não suportados.
    ValueError
        Se os arrays estiverem vazios, não forem unidimensionais,
        tiverem formas diferentes, ou contiverem valores NaN/Inf.
    """

    # Teste de X e Y sendo arrays NumPy.
    if not isinstance(X, np.ndarray):
        raise TypeError("X precisa ser um array NumPy.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred precisa ser um array NumPy.")

    # Verificar dimensão dos array, devem ser unidimensionais
    if X.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Entradas devem ser arrays unidimensionais.")

    # Entrada não pode ser vazia
    if X.size == 0 or y_pred.size == 0:
        raise ValueError("Entradas não podem ser arrays vazios.")

    # Entradas precisam ter o mesmo tamanho
    if X.shape != y_pred.shape:
        raise ValueError("Entradas precisam ter o mesmo tamanho")

    # Verifica se X contém números reais
    if not (np.issubdtype(X.dtype, np.integer) or np.issubdtype(X.dtype, np.floating)):
        raise TypeError("X deve conter valores numéricos reais.")

    # Verifica se y_pred contém números reais
    if not (np.issubdtype(y_pred.dtype, np.integer)or np.issubdtype(y_pred.dtype, np.floating)):
        raise TypeError("Y deve conter valores numéricos reais.")

    # Verifica se não existem NaN ou infinitos
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Entradas não podem conter valores NaN ou infinitos.")

    # Padroniza dados válidos como float64
    X = X.astype(np.float64, copy=False) #copy false para não criar uma cópia desnecessária se já for float64
    y_pred = y_pred.astype(np.float64, copy=False)
    return X, y_pred

