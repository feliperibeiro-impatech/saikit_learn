import numpy as np
import pytest

from saikit_learn.neighbors import KNN


# DADOS DE TESTE

# Dois blocos bem separados, usados em vários testes simples.
@pytest.fixture
def dados_simples():
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [8, 8],
        [9, 9],
        [10, 10],
    ], dtype=float)
    y = np.array(["A", "A", "A", "B", "B", "B"])
    return X, y


# CORRETUDE DO CLASSIFICADOR

# Testa que o KNN classifica corretamente pontos próximos de cada grupo.
def test_classifica_pontos_simples(dados_simples):
    X, y = dados_simples
    modelo = KNN(k=3).fit(X, y)
    previsto = modelo.predict(np.array([
        [2, 2],
        [9, 9],
    ], dtype=float))
    assert list(previsto) == ["A", "B"]


# Testa o funcionamento com k = 1.
def test_k_um(dados_simples):
    X, y = dados_simples
    modelo = KNN(k=1).fit(X, y)
    previsto = modelo.predict(np.array([
        [1.1, 1.1],
        [9.1, 9.1],
    ], dtype=float))
    assert list(previsto) == ["A", "B"]


# Testa que o modelo acerta as próprias amostras quando k = 1.
def test_acerta_amostras_de_treino_com_k_um(dados_simples):
    X, y = dados_simples
    modelo = KNN(k=1).fit(X, y)
    previsto = modelo.predict(X)
    assert np.array_equal(previsto, y)


# Testa o funcionamento com três classes.
def test_tres_classes():
    X = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [10, 0],
        [11, 0],
        [10, 1],
        [0, 10],
        [1, 10],
        [0, 11],
    ], dtype=float)
    y = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])

    modelo = KNN(k=3).fit(X, y)

    previsto = modelo.predict(np.array([
        [0.2, 0.2],
        [10.2, 0.2],
        [0.2, 10.2],
    ], dtype=float))

    assert list(previsto) == ["A", "B", "C"]


# MÉTRICAS DE DISTÂNCIA

# Testa a distância Manhattan, correspondente a p = 1.
def test_distancia_manhattan():
    X = np.array([
        [0, 2],
        [2, 0],
    ], dtype=float)
    y = np.array(["A", "B"])

    modelo = KNN(k=1, p=1).fit(X, y)

    previsto = modelo.predict(np.array([
        [0, 0],
    ], dtype=float))

    assert previsto[0] == "A"


# Testa a distância Euclidiana, correspondente a p = 2.
def test_distancia_euclidiana():
    X = np.array([
        [0, 2],
        [3, 0],
    ], dtype=float)
    y = np.array(["A", "B"])

    modelo = KNN(k=1, p=2).fit(X, y)

    previsto = modelo.predict(np.array([
        [0, 0],
    ], dtype=float))

    assert previsto[0] == "A"


# DESMPATE

# Testa que, em caso de empate na quantidade de votos, vence o rótulo
# cuja primeira ocorrência está mais próxima.
def test_desempate_pelo_vizinho_mais_proximo():
    X = np.array([
        [0],
        [1],
        [3],
        [4],
    ], dtype=float)
    y = np.array(["A", "B", "B", "A"])

    modelo = KNN(k=4).fit(X, y)

    previsto = modelo.predict(np.array([
        [0.1],
    ], dtype=float))

    assert previsto[0] == "A"


# RÓTULOS E FORMATO DA SAÍDA

# Testa que o KNN funciona com diferentes tipos de rótulos.
@pytest.mark.parametrize("rotulos", [
    ("A", "B"),
    (0, 1),
    (1, 2),
    (-1, 1),
])
def test_rotulos_de_tipos_diferentes(rotulos):
    r_a, r_b = rotulos

    X = np.array([
        [0, 0],
        [1, 1],
        [10, 10],
        [11, 11],
    ], dtype=float)

    y = np.array([r_a, r_a, r_b, r_b])

    modelo = KNN(k=1).fit(X, y)

    previsto = modelo.predict(np.array([
        [0.2, 0.2],
        [10.2, 10.2],
    ], dtype=float))

    assert list(previsto) == [r_a, r_b]


# Testa o formato da saída.
def test_formato_da_saida(dados_simples):
    X, y = dados_simples
    modelo = KNN(k=3).fit(X, y)

    previsto = modelo.predict(X[:4])

    assert isinstance(previsto, np.ndarray)
    assert previsto.shape == (4,)
    assert set(previsto) <= set(y)


# ESTADO DO MODELO

# Testa que fit devolve o próprio objeto.
def test_fit_retorna_self(dados_simples):
    X, y = dados_simples
    modelo = KNN(k=3)

    assert modelo.fit(X, y) is modelo


# Testa que um novo fit substitui os dados anteriores.
def test_refit_substitui_dados_antigos():
    X1 = np.array([
        [0, 0],
        [1, 1],
    ], dtype=float)
    y1 = np.array(["A", "A"])

    X2 = np.array([
        [10, 10],
        [11, 11],
    ], dtype=float)
    y2 = np.array(["B", "B"])

    modelo = KNN(k=1).fit(X1, y1)
    modelo.fit(X2, y2)

    previsto = modelo.predict(np.array([
        [10.1, 10.1],
    ], dtype=float))

    assert previsto[0] == "B"


# VALIDAÇÕES DO CONSTRUTOR

# Testa que k não pode ser menor que 1.
def test_k_menor_que_um():
    with pytest.raises(ValueError):
        KNN(k=0)


# Testa que p deve ser 1 ou 2.
@pytest.mark.parametrize("p", [0, 3, -1])
def test_p_invalido(p):
    with pytest.raises(ValueError):
        KNN(k=1, p=p)


# VALIDAÇÕES DE ENTRADA

# Testa que fit rejeita X que não seja array do NumPy.
def test_fit_rejeita_X_que_nao_e_array():
    X = [[1, 2], [3, 4]]
    y = np.array(["A", "B"])

    with pytest.raises(TypeError):
        KNN(k=1).fit(X, y)


# Testa que fit rejeita y que não seja array do NumPy.
def test_fit_rejeita_y_que_nao_e_array():
    X = np.array([
        [1, 2],
        [3, 4],
    ], dtype=float)
    y = ["A", "B"]

    with pytest.raises(TypeError):
        KNN(k=1).fit(X, y)


# Testa que fit rejeita X e y com tamanhos diferentes.
def test_fit_rejeita_tamanhos_incompativeis():
    X = np.array([
        [1, 2],
        [3, 4],
    ], dtype=float)
    y = np.array(["A"])

    with pytest.raises(ValueError):
        KNN(k=1).fit(X, y)


# Testa que predict rejeita entrada que não seja array do NumPy.
def test_predict_rejeita_entrada_que_nao_e_array(dados_simples):
    X, y = dados_simples
    modelo = KNN(k=3).fit(X, y)

    with pytest.raises(TypeError):
        modelo.predict([[1, 2]])