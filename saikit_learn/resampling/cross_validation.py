import numpy as np


def cross_validation(model, X, y, scoring, k=5, shuffle=True, random_state=None):
    """Avalia um modelo usando validação cruzada K-Fold.
    Quando k é igual ao número de amostras, a função vira um Leave-One-Out Cross Validation (LOOCV).

    Por padrão, as amostras são embaralhadas antes de serem divididas em folds,
    o que reduz folds com classes ausentes quando os dados estão ordenados por
    classe. Use shuffle=False quando a ordem das amostras importar (como em séries temporais). O modelo
    recebido é ajustado a cada fold e termina treinado no último deles.

    Args:
        model: Modelo com os métodos fit(X, y) e predict(X).
        X (np.ndarray): Matriz de atributos.
        y (np.ndarray): Vetor de valores verdadeiros.
        scoring (callable): Função que recebe os valores verdadeiros e as
            previsões e retorna um score.
        k (int): Número de folds. Deve satisfazer
            2 <= k <= número de amostras.
        shuffle (bool): Se True (padrão), embaralha as amostras antes de
            dividi-las em folds. Se False, os folds são faixas contíguas.
        random_state (int | None): Semente do embaralhamento. Use um inteiro
            não negativo para resultados reproduzíveis (e para comparar
            modelos nos mesmos folds). Só é usado se shuffle=True.

    Returns:
        np.ndarray: Array contendo o score obtido em cada fold.
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X deve ser um np.ndarray.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y deve ser um np.ndarray.")

    if X.ndim != 2:
        raise ValueError("X deve ser uma matriz 2D.")

    if y.ndim != 1:
        raise ValueError("y deve ser um vetor 1D.")

    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo número de amostras.")

    if len(X) == 0:
        raise ValueError("X e y não podem estar vazios.")

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k deve ser um inteiro.")

    if not callable(scoring):
        raise TypeError("scoring deve ser uma função.")

    if not isinstance(shuffle, (bool, np.bool_)):
        raise TypeError("shuffle deve ser um booleano.")

    if random_state is not None:
        if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
            raise TypeError("random_state deve ser None ou um inteiro.")

        if random_state < 0:
            raise ValueError("random_state deve ser não negativo.")

    n_samples = len(X)

    if k < 2 or k > n_samples:
        raise ValueError("k deve satisfazer 2 <= k <= o número de amostras.")

    fold_sizes = np.full(k, n_samples // k, dtype=int)

    fold_sizes[:n_samples % k] += 1

    indices = np.arange(n_samples)

    if shuffle:
        np.random.default_rng(random_state).shuffle(indices)

    scores = np.empty(k)

    start = 0

    for i, fold_size in enumerate(fold_sizes):
        stop = start + fold_size

        test_indices = indices[start:stop]

        train_indices = np.concatenate((indices[:start], indices[stop:]))

        X_train = X[train_indices]
        y_train = y[train_indices]

        X_test = X[test_indices]
        y_test = y[test_indices]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        scores[i] = scoring(y_test, predictions)

        start = stop

    return scores