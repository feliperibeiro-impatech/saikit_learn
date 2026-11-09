import numpy as np
import pytest

from saikit_learn.discriminant_analysis import QDA


# DADOS DE TESTE

# Duas classes com a mesma média (0, 0) e covariâncias diferentes, separáveis só por fronteira quadrática.
@pytest.fixture
def dados_quadraticos():
    rng = np.random.default_rng(0)
    n = 200
    X_a = rng.normal(0, 1, size=(n, 2)) * np.array([0.5, 3.0])
    X_b = rng.normal(0, 1, size=(n, 2)) * np.array([3.0, 0.5])
    X = np.vstack([X_a, X_b])
    Y = np.array(["A"] * n + ["B"] * n)
    return X, Y


# Dois blocos bem separados, usados em vários testes simples.
@pytest.fixture
def dados_simples():
    X = np.array([
        [1, 2], [2, 3], [3, 3], [2, 1],
        [8, 8], [9, 7], [10, 9], [9, 10],
    ], dtype=float)
    Y = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    return X, Y


# CORRETUDE DO CLASSIFICADOR

# Testa que o QDA separa classes com a mesma média e covariâncias diferentes.
def test_fronteira_quadratica_mesma_media(dados_quadraticos):
    X, Y = dados_quadraticos
    modelo = QDA().fit(X, Y)
    previsto = modelo.predict(np.array([[4, 0], [-4, 0], [0, 4], [0, -4]]))
    assert list(previsto) == ["B", "B", "A", "A"]


# Testa um caso unidimensional com resposta conferível à mão.
def test_caso_unidimensional_conferido_a_mao():
    X = np.array([[-1], [0], [1], [-4], [0], [4]])
    Y = np.array(["A", "A", "A", "B", "B", "B"])
    modelo = QDA().fit(X, Y)
    assert list(modelo.predict(np.array([[0.5], [6], [-6]]))) == ["A", "B", "B"]


# Testa que o modelo acerta as próprias amostras de treino quando as classes estão bem separadas.
def test_acerta_amostras_de_treino(dados_simples):
    X, Y = dados_simples
    modelo = QDA().fit(X, Y)
    assert np.array_equal(modelo.predict(X), Y)


# Testa que o argmax funciona com três classes.
def test_tres_classes():
    rng = np.random.default_rng(1)
    centros = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    X = np.vstack([c + rng.normal(0, 1, size=(30, 2)) for c in centros])
    Y = np.repeat(["A", "B", "C"], 30)
    modelo = QDA().fit(X, Y)
    previsto = modelo.predict(np.array([[0.2, -0.1], [9.8, 0.3], [-0.2, 10.1]]))
    assert list(previsto) == ["A", "B", "C"]


# RÓTULOS E FORMATO DA SAÍDA

# Testa que o resultado independe do tipo e dos valores dos rótulos, pegando o erro de índice k versus rótulo c.
@pytest.mark.parametrize("rotulos", [
    ("A", "B"),
    (0, 1),
    (1, 2),
    (-1, 1),
    (10, 3),
])
def test_rotulos_de_tipos_diferentes(dados_quadraticos, rotulos):
    X, Y_txt = dados_quadraticos
    r_a, r_b = rotulos
    Y = np.where(Y_txt == "A", r_a, r_b)
    modelo = QDA().fit(X, Y)
    previsto = modelo.predict(np.array([[4, 0], [0, 4]]))
    assert list(previsto) == [r_b, r_a]


# Testa que a saída é um vetor 1D com uma previsão por linha de X, usando só rótulos do treino.
def test_formato_da_saida(dados_simples):
    X, Y = dados_simples
    modelo = QDA().fit(X, Y)
    previsto = modelo.predict(X[:5])
    assert isinstance(previsto, np.ndarray)
    assert previsto.shape == (5,)
    assert set(previsto) <= set(Y)


# Testa que o predict devolve o rótulo original e não o índice da classe.
def test_predict_devolve_rotulo_e_nao_indice(dados_simples):
    X, Y = dados_simples
    modelo = QDA().fit(X, Y)
    previsto = modelo.predict(np.array([[9, 9]]))
    assert previsto[0] == "B"
    assert not isinstance(previsto[0], (int, np.integer))


# Testa que uma amostra 1D como [2, 2] é aceita e tratada como matriz (1, d).
def test_predict_com_amostra_1d(dados_simples):
    X, Y = dados_simples
    modelo = QDA().fit(X, Y)
    previsto = modelo.predict(np.array([2, 2]))
    assert previsto.shape == (1,)
    assert previsto[0] == "A"



# ESTADO DO MODELO

# Testa que o fit devolve o próprio objeto, permitindo encadear chamadas.
def test_fit_retorna_self(dados_simples):
    X, Y = dados_simples
    modelo = QDA()
    assert modelo.fit(X, Y) is modelo


# Testa que embaralhar as amostras de treino não altera as previsões.
def test_ordem_das_amostras_nao_importa(dados_quadraticos):
    X, Y = dados_quadraticos
    perm = np.random.default_rng(2).permutation(len(Y))
    pontos = np.array([[4, 0], [0, 4], [-3, 0.2], [0.1, -3]])
    original = QDA().fit(X, Y).predict(pontos)
    embaralhado = QDA().fit(X[perm], Y[perm]).predict(pontos)
    assert np.array_equal(original, embaralhado)


# Testa que um segundo fit descarta as classes do primeiro.
def test_refit_descarta_classes_antigas(dados_simples):
    X, Y = dados_simples
    modelo = QDA().fit(X, Y)

    rng = np.random.default_rng(3)
    X2 = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(6, 1, (20, 2))])
    Y2 = np.array([7] * 20 + [8] * 20)
    modelo.fit(X2, Y2)

    previsto = modelo.predict(X2)
    assert set(previsto) <= {7, 8}


# VALIDAÇÕES DE ENTRADA

# Testa que predict antes do fit levanta RuntimeError.
def test_predict_antes_do_fit():
    with pytest.raises(RuntimeError):
        QDA().predict(np.array([[1, 2]]))


# Testa que o predict rejeita um número de colunas diferente do usado no fit.
def test_predict_com_numero_errado_de_colunas(dados_simples):
    X, Y = dados_simples
    modelo = QDA().fit(X, Y)
    with pytest.raises(ValueError):
        modelo.predict(np.array([[1, 2, 3]]))


# Testa que o fit rejeita um X que não é matriz 2D.
def test_fit_com_X_nao_2d():
    with pytest.raises(ValueError):
        QDA().fit(np.array([1, 2, 3, 4]), np.array(["A", "A", "B", "B"]))


# Testa que o fit rejeita X e Y com números de amostras diferentes.
def test_fit_com_tamanhos_incompativeis(dados_simples):
    X, Y = dados_simples
    with pytest.raises(ValueError):
        QDA().fit(X, Y[:-1])


# Testa que o fit rejeita um Y que não é vetor 1D.
def test_fit_com_Y_nao_1d(dados_simples):
    X, Y = dados_simples
    with pytest.raises(ValueError):
        QDA().fit(X, Y.reshape(-1, 1))


# Testa que o fit rejeita NaN em X.
def test_fit_com_nan(dados_simples):
    X, Y = dados_simples
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        QDA().fit(X, Y)


# Testa que o fit rejeita o caso com tantas amostras quanto classes (ou menos).
def test_fit_com_amostras_insuficientes():
    with pytest.raises(ValueError):
        QDA().fit(np.array([[1, 2], [3, 4]]), np.array(["A", "B"]))


# CASOS SENSÍVEIS ESPECÍFICOS DO QDA

# Testa que uma classe com uma única amostra gera erro explícito, e não resultado com NaN.
def test_classe_com_uma_unica_amostra():
    X = np.array([[1, 2], [2, 3], [3, 1], [9, 9]])
    Y = np.array(["A", "A", "A", "B"])
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        QDA().fit(X, Y).predict(np.array([[1, 1]]))


# Testa que uma classe com menos amostras do que features (covariância singular) gera erro explícito.
def test_classe_com_covariancia_singular():
    rng = np.random.default_rng(4)
    X_a = rng.normal(0, 1, size=(20, 3))
    X_b = rng.normal(5, 1, size=(2, 3))  # 2 amostras em 3 dimensões
    X = np.vstack([X_a, X_b])
    Y = np.array(["A"] * 20 + ["B"] * 2)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        QDA().fit(X, Y).predict(np.array([[0, 0, 0]]))


# Testa que features colineares (colunas idênticas) geram erro explícito.
def test_features_colineares():
    rng = np.random.default_rng(5)
    col = rng.normal(0, 1, size=(30, 1))
    X = np.hstack([col, col])  # segunda coluna é cópia da primeira
    X[15:] += 5
    Y = np.array(["A"] * 15 + ["B"] * 15)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        QDA().fit(X, Y).predict(np.array([[0, 0]]))


# Testa que o prior entra na decisão, levando um ponto equidistante para a classe majoritária.
def test_prior_desempata_ponto_equidistante():
    X_a = np.array([[-1], [-1], [0], [1], [1]])
    X_b = np.array([[9], [10], [11]])
    X = np.vstack([X_a, X_b])
    Y = np.array(["A"] * len(X_a) + ["B"] * len(X_b))
    modelo = QDA().fit(X, Y)
    assert modelo.predict(np.array([[5]]))[0] == "A"
