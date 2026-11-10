import numpy as np

def bootstrap(data : np.ndarray, n_resamples : int, sample_size = 0) -> np.ndarray:
    """Método de bootstrap para reamostra

    Args:
        data (np.ndarray): Amostra
        n_resamples (int): Número de reamostras
        sample_size (int, opcional): Tamanho das reamostras. Padroniza para 0.

    Raises:
        TypeError: Número de reamostras precisa ser um inteiro
        TypeError: Tamanho da amostra precisa ser um inteiro
        ValueError: Número de reamostras precisa ser positivo

    Returns:
        np.ndarray: Array com reamostras
    """
    if not isinstance(n_resamples, (int, np.integer)):
        raise TypeError("Número de reamostras precisa ser um inteiro")
    if not isinstance(sample_size, (int, np.integer)):
        raise TypeError("Tamanho da amostra precisa ser um inteiro")
    if n_resamples < 1:
        raise ValueError("Número de reamostras precisa ser positivo")
    if sample_size < 1:
        sample_size = np.size(data)
    return np.random.choice(data, (n_resamples, sample_size))