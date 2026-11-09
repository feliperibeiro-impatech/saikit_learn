import numpy as np
import pytest

from saikit_learn.discriminant_analysis import LDA


# DADOS DE TESTE

# Duas classes com a mesma covariância e médias diferentes, o caso ideal para o LDA.
@pytest.fixture
def dados_lineares():
    rng = np.random.default_rng(0)
    n = 200
    X_a = rng.normal(0, 1, size=(n, 2)) * np.array([1.0, 2.0])
    X_b = rng.normal(0, 1, size=(n, 2)) * np.array([1.0, 2.0]) + np.array([6.0, 0.0])
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


# Caso unidimensional: A = {-1, 0, 1} e B = {3, 4, 5}, com covariância agrupada igual a 1.
@pytest.fixture
def dados_1d():
    X = np.array([[-1], [0], [1], [3], [4], [5]])
    Y = np.array(["A", "A", "A", "B", "B", "B"])
    return X, Y


# CORRETUDE DO CLASSIFICADOR

# Testa que o LDA separa duas classes gaussianas com a mesma covariância.
def test_separa_classes_com_mesma_covariancia(dados_lineares):
    X, Y = dados_lineares
    modelo = LDA().fit(X, Y)
    previsto = modelo.predict(np.array([[0, 0], [6, 0], [-1, 1], [7, -1]]))
    assert list(previsto) == ["A", "B", "A", "B"]


# Testa um caso unidimensional cuja fronteira, em x = 2, se confere à mão.
def test_caso_unidimensional_conferido_a_mao(dados_1d):
    X, Y = dados_1d
    modelo = LDA().fit(X, Y)
    assert list(modelo.predict(np.array([[1.5], [2.5], [-10], [10]]))) == ["A", "B", "A", "B"]


# Testa que o modelo acerta as próprias amostras de treino quando as classes estão bem separadas.
def test_acerta_amostras_de_treino(dados_simples):
    X, Y = dados_simples
    modelo = LDA().fit(X, Y)
    assert np.array_equal(modelo.predict(X), Y)


# Testa que o argmax funciona com três classes.
def test_tres_classes():
    rng = np.random.default_rng(1)
    centros = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    X = np.vstack([c + rng.normal(0, 1, size=(30, 2)) for c in centros])
    Y = np.repeat(["A", "B", "C"], 30)
    modelo = LDA().fit(X, Y)
    previsto = modelo.predict(np.array([[0.2, -0.1], [9.8, 0.3], [-0.2, 10.1]]))
    assert list(previsto) == ["A", "B", "C"]


# Testa que o prior entra na decisão, levando um ponto equidistante para a classe majoritária.
def test_prior_desempata_ponto_equidistante():
    base = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=float)
    X = np.vstack([np.vstack([base] * 9), base + 10])
    Y = np.array(["A"] * 45 + ["B"] * 5)
    assert LDA().fit(X, Y).predict(np.array([[5, 5]]))[0] == "A"


# Testa o mesmo cenário com a classe majoritária trocada, garantindo que o desempate vem do prior e não da ordem dos rótulos.
def test_prior_desempata_para_a_outra_classe():
    base = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=float)
    X = np.vstack([base, np.vstack([base + 10] * 9)])
    Y = np.array(["A"] * 5 + ["B"] * 45)
    assert LDA().fit(X, Y).predict(np.array([[5, 5]]))[0] == "B"


# ESTIMAÇÃO DOS PARÂMETROS

# Testa que médias e priors estimados batem com os valores calculados à mão.
def test_medias_e_priors_estimados(dados_1d):
    X, Y = dados_1d
    modelo = LDA().fit(X, Y)
    assert np.allclose(modelo._means, [[0.0], [4.0]])
    assert np.allclose(modelo._priors, [0.5, 0.5])


# Testa que a covariância agrupada vale 1 no caso unidimensional, conferido à mão com denominador n - K.
def test_covariancia_agrupada_conferida_a_mao(dados_1d):
    X, Y = dados_1d
    modelo = LDA().fit(X, Y)
    assert np.allclose(modelo._covariance, [[1.0]])


# Testa que a covariância agrupada é simétrica e tem shape (d, d).
def test_covariancia_simetrica(dados_lineares):
    X, Y = dados_lineares
    S = LDA().fit(X, Y)._covariance
    assert S.shape == (2, 2)
    assert np.allclose(S, S.T)


# Testa que os priors somam 1.
def test_priors_somam_um(dados_lineares):
    X, Y = dados_lineares
    assert np.isclose(LDA().fit(X, Y)._priors.sum(), 1.0)


# Testa que a covariância é centralizada por classe, e portanto não muda ao transladar uma classe inteira.
def test_covariancia_ignora_diferenca_entre_medias(dados_lineares):
    X, Y = dados_lineares
    X_transladado = X.copy()
    X_transladado[Y == "B"] += 100
    S1 = LDA().fit(X, Y)._covariance
    S2 = LDA().fit(X_transladado, Y)._covariance
    assert np.allclose(S1, S2)


# RÓTULOS E FORMATO DA SAÍDA

# Testa que o resultado independe do tipo e dos valores dos rótulos, pegando o erro de índice k versus rótulo c.
@pytest.mark.parametrize("rotulos", [
    ("A", "B"),
    (0, 1),
    (1, 2),
    (-1, 1),
    (10, 3),
])
def test_rotulos_de_tipos_diferentes(dados_lineares, rotulos):
    X, Y_txt = dados_lineares
    r_a, r_b = rotulos
    Y = np.where(Y_txt == "A", r_a, r_b)
    previsto = LDA().fit(X, Y).predict(np.array([[0, 0], [6, 0]]))
    assert list(previsto) == [r_a, r_b]


# Testa que a saída é um vetor 1D com uma previsão por linha de X, usando só rótulos do treino.
def test_formato_da_saida(dados_simples):
    X, Y = dados_simples
    previsto = LDA().fit(X, Y).predict(X[:5])
    assert isinstance(previsto, np.ndarray)
    assert previsto.shape == (5,)
    assert set(previsto) <= set(Y)


# Testa que o predict devolve o rótulo original e não o índice da classe.
def test_predict_devolve_rotulo_e_nao_indice(dados_simples):
    X, Y = dados_simples
    previsto = LDA().fit(X, Y).predict(np.array([[9, 9]]))
    assert previsto[0] == "B"
    assert not isinstance(previsto[0], (int, np.integer))


# Testa que uma amostra 1D como [2, 2] é aceita e tratada como matriz (1, d).
def test_predict_com_amostra_1d(dados_simples):
    X, Y = dados_simples
    previsto = LDA().fit(X, Y).predict(np.array([2, 2]))
    assert previsto.shape == (1,)
    assert previsto[0] == "A"



# ESTADO DO MODELO

# Testa que o fit devolve o próprio objeto, permitindo encadear chamadas.
def test_fit_retorna_self(dados_simples):
    X, Y = dados_simples
    modelo = LDA()
    assert modelo.fit(X, Y) is modelo


# Testa que embaralhar as amostras de treino não altera as previsões.
def test_ordem_das_amostras_nao_importa(dados_lineares):
    X, Y = dados_lineares
    perm = np.random.default_rng(2).permutation(len(Y))
    pontos = np.array([[0, 0], [6, 0], [3.5, 1], [2.5, -1]])
    original = LDA().fit(X, Y).predict(pontos)
    embaralhado = LDA().fit(X[perm], Y[perm]).predict(pontos)
    assert np.array_equal(original, embaralhado)


# Testa que um segundo fit descarta as classes do primeiro.
def test_refit_descarta_classes_antigas(dados_simples):
    X, Y = dados_simples
    modelo = LDA().fit(X, Y)

    rng = np.random.default_rng(3)
    X2 = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(6, 1, (20, 2))])
    Y2 = np.array([7] * 20 + [8] * 20)
    modelo.fit(X2, Y2)

    assert set(modelo.predict(X2)) <= {7, 8}
    assert len(modelo._classes) == 2


# VALIDAÇÕES DE ENTRADA

# Testa que predict antes do fit levanta RuntimeError.
def test_predict_antes_do_fit():
    with pytest.raises(RuntimeError):
        LDA().predict(np.array([[1, 2]]))


# Testa que o predict rejeita um número de colunas diferente do usado no fit.
def test_predict_com_numero_errado_de_colunas(dados_simples):
    X, Y = dados_simples
    modelo = LDA().fit(X, Y)
    with pytest.raises(ValueError):
        modelo.predict(np.array([[1, 2, 3]]))


# Testa que o fit rejeita um X que não é matriz 2D.
def test_fit_com_X_nao_2d():
    with pytest.raises(ValueError):
        LDA().fit(np.array([1, 2, 3, 4]), np.array(["A", "A", "B", "B"]))


# Testa que o fit rejeita X e Y com números de amostras diferentes.
def test_fit_com_tamanhos_incompativeis(dados_simples):
    X, Y = dados_simples
    with pytest.raises(ValueError):
        LDA().fit(X, Y[:-1])


# Testa que o fit rejeita um Y que não é vetor 1D.
def test_fit_com_Y_nao_1d(dados_simples):
    X, Y = dados_simples
    with pytest.raises(ValueError):
        LDA().fit(X, Y.reshape(-1, 1))


# Testa que o fit rejeita NaN em X.
def test_fit_com_nan(dados_simples):
    X, Y = dados_simples
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        LDA().fit(X, Y)


# Testa que o fit rejeita o caso com tantas amostras quanto classes (ou menos), pois n - K zeraria o denominador.
def test_fit_com_amostras_insuficientes():
    with pytest.raises(ValueError):
        LDA().fit(np.array([[1, 2], [3, 4]]), np.array(["A", "B"]))


# CASOS SENSÍVEIS ESPECÍFICOS DO LDA

# Testa que uma classe com uma única amostra é válida no LDA, pois a covariância é agrupada entre as classes.
def test_classe_com_uma_unica_amostra_e_valida():
    X = np.array([[1, 2], [2, 3], [3, 1], [9, 9]])
    Y = np.array(["A", "A", "A", "B"])
    previsto = LDA().fit(X, Y).predict(np.array([[1, 1], [9, 9]]))
    assert list(previsto) == ["A", "B"]


# Testa que d > n - K (covariância agrupada singular) gera erro explícito, e não previsões instáveis.
def test_covariancia_singular_por_poucas_amostras():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, size=(6, 5))  # n - K = 4 < d = 5
    X[3:] += 5
    Y = np.array(["A"] * 3 + ["B"] * 3)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        LDA().fit(X, Y).predict(np.array([[0, 0, 0, 0, 0]]))


# Testa que features colineares (colunas idênticas) geram erro explícito.
def test_features_colineares():
    rng = np.random.default_rng(5)
    col = rng.normal(0, 1, size=(30, 1))
    X = np.hstack([col, col])  # segunda coluna é cópia da primeira
    X[15:] += 5
    Y = np.array(["A"] * 15 + ["B"] * 15)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        LDA().fit(X, Y).predict(np.array([[0, 0]]))


# Testa que uma feature constante (variância zero) gera erro explícito.
def test_feature_constante():
    X = np.array([[1, 7], [2, 7], [3, 7], [8, 7], [9, 7], [10, 7]], dtype=float)
    Y = np.array(["A", "A", "A", "B", "B", "B"])
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        LDA().fit(X, Y).predict(np.array([[2, 7]]))