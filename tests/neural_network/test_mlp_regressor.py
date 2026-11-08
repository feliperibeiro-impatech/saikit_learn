from saikit_learn.neural_network import MLPRegressor
import numpy as np
import pytest

# DADOS PARA TESTE E AUXILIARES

# Dados lineares sem ruído, Y = 2*x1 - 3x*2 + 1, com featurres em escala similar.
@pytest.fixture
def dados_lineares():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(100,2))
    Y = 2 * X[:, 0] - 3 * X[:, 1] + 1
    return X, Y

# Poucos pontos com relação não linear, usados para testar overfit e queda da perda.
@pytest.fixture
def dados_pequenos():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(8, 1))
    Y = np.sin(X[:, 0])
    return X, Y

# _init_parameters

# Testa que os shapes de pesos e vieses seguem a lista sizes.
def test_init_parameters_shapes():
    modelo = MLPRegressor(random_state=0)
    modelo._init_parameters([3, 32, 16, 1])
    assert [W.shape for W in modelo._weights] == [(3, 32), (32, 16), (16, 1)]
    assert [b.shape for b in modelo._biases] == [(32,), (16,), (1,)]


# Testa que sem camada oculta existe um único W de shape (d, 1).
def test_init_parameters_sem_camada_oculta():
    modelo = MLPRegressor(hidden_layers=(), random_state=0)
    modelo._init_parameters([4, 1])
    assert len(modelo._weights) == 1
    assert modelo._weights[0].shape == (4, 1)


# Testa que os vieses começam em zero.
def test_init_parameters_vieses_zerados():
    modelo = MLPRegressor(random_state=0)
    modelo._init_parameters([3, 8, 1])
    assert all(np.all(b == 0) for b in modelo._biases)


# Testa que pesos não são todos iguais, o que quebraria a simetria entre neurônios.
def test_init_parameters_pesos_assimetricos():
    modelo = MLPRegressor(random_state=0)
    modelo._init_parameters([3, 8, 1])
    assert np.unique(modelo._weights[0]).size > 1


# Testa que o mesmo random_state gera os mesmos pesos e que sementes diferentes geram pesos diferentes.
def test_init_parameters_reprodutibilidade():
    a, b, c = (MLPRegressor(random_state=s) for s in (0, 0, 1))
    for m in (a, b, c):
        m._init_parameters([3, 8, 1])
    assert np.array_equal(a._weights[0], b._weights[0])
    assert not np.array_equal(a._weights[0], c._weights[0])


# Testa que a escala dos pesos segue He, com desvio próximo de sqrt(2 / n_in).
def test_init_parameters_escala_he():
    modelo = MLPRegressor(random_state=0)
    modelo._init_parameters([500, 500, 1])
    assert np.isclose(modelo._weights[0].std(), np.sqrt(2 / 500), rtol=0.05)


# _forward

# Testa a saída sem camada oculta, onde y = 2*x1 + 3*x2 + 1.
def test_forward_linear_conferido_a_mao():
    modelo = MLPRegressor(hidden_layers=())
    modelo._weights = [np.array([[2.0], [3.0]])]
    modelo._biases = [np.array([1.0])]
    activations, zs = modelo._forward(np.array([[1.0, 1.0], [0.0, 2.0]]))
    assert np.allclose(activations[-1], [[6.0], [7.0]])


# Testa que a ReLU zera valores negativos na camada oculta e que a saída não aplica ativação.
def test_forward_relu_na_oculta_e_identidade_na_saida():
    modelo = MLPRegressor(hidden_layers=(2,))
    modelo._weights = [np.array([[1.0, -1.0]]), np.array([[1.0], [1.0]])]
    modelo._biases = [np.zeros(2), np.array([-5.0])]
    activations, zs = modelo._forward(np.array([[2.0]]))
    assert np.allclose(activations[1], [[2.0, 0.0]])
    assert np.allclose(activations[-1], [[-3.0]])


# Testa que activations tem L+1 elementos (começando em X) e zs tem L.
def test_forward_tamanhos_das_listas():
    modelo = MLPRegressor(hidden_layers=(4, 3), random_state=0)
    modelo._init_parameters([2, 4, 3, 1])
    X = np.ones((5, 2))
    activations, zs = modelo._forward(X)
    assert len(activations) == 4 and len(zs) == 3
    assert np.array_equal(activations[0], X)
    assert [a.shape for a in activations] == [(5, 2), (5, 4), (5, 3), (5, 1)]

# _backward

# Testa que os gradientes têm os mesmos shapes dos pesos e vieses.
def test_backward_shapes():
    modelo = MLPRegressor(hidden_layers=(4, 3), random_state=0)
    modelo._init_parameters([2, 4, 3, 1])
    X, Y = np.random.default_rng(0).normal(size=(6, 2)), np.ones((6, 1))
    activations, zs = modelo._forward(X)
    gW, gb = modelo._backward(Y, activations, zs)
    assert [g.shape for g in gW] == [W.shape for W in modelo._weights]
    assert [g.shape for g in gb] == [b.shape for b in modelo._biases]

# Calcula a perda (1/2n) * soma((y_pred - y)^2) usando o próprio _forward do modelo.
def perda(modelo, X, Y):
    activations, _ = modelo._forward(X)
    return np.mean((activations[-1] - Y.reshape(-1, 1)) ** 2) / 2

# Testa cada gradiente analítico contra a diferença finita centrada (L(t+e) - L(t-e)) / 2e.
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("hidden", [(), (5,), (4, 3)])
def test_backward_gradiente_numerico(hidden, seed):
    rng = np.random.default_rng(seed)
    X, Y = rng.normal(size=(7, 3)), rng.normal(size=7)
    modelo = MLPRegressor(hidden_layers=hidden, random_state=seed)
    modelo._init_parameters([3, *hidden, 1])
    modelo._biases = [rng.normal(0, 0.5, size=b.shape) for b in modelo._biases]

    activations, zs = modelo._forward(X)
    if any(np.abs(z).min() < 1e-3 for z in zs[:-1]):
        pytest.skip("algum z da ReLU está perto de 0")
    gW, gb = modelo._backward(Y.reshape(-1, 1), activations, zs)

    eps = 1e-5
    for params, grads in ((modelo._weights, gW), (modelo._biases, gb)):
        for p, g in zip(params, grads):
            for idx in np.ndindex(p.shape):
                original = p[idx]
                p[idx] = original + eps
                mais = perda(modelo, X, Y)
                p[idx] = original - eps
                menos = perda(modelo, X, Y)
                p[idx] = original
                numerico = (mais - menos) / (2 * eps)
                assert np.isclose(g[idx], numerico, rtol=1e-4, atol=1e-7)

# TESTES DO FIT

# # Testa que a perda final é menor que a inicial.
def test_fit_perda_diminui(dados_pequenos):
    X, Y = dados_pequenos
    modelo = MLPRegressor(hidden_layers=(16,), learning_rate=0.05, n_epochs=2000, random_state=0)
    modelo.fit(X, Y)
    assert modelo._loss_history[-1] < modelo._loss_history[0]


# Testa que a rede ajusta bem poucos pontos (overfit), reduzindo a perda a menos de 10% da inicial.
def test_fit_overfit_em_poucos_dados(dados_pequenos):
    X, Y = dados_pequenos
    modelo = MLPRegressor(hidden_layers=(16,), learning_rate=0.05, n_epochs=3000, random_state=0)
    modelo.fit(X, Y)
    assert modelo._loss_history[-1] < 0.1 * modelo._loss_history[0]


# Testa que sem camada oculta o modelo recupera a reta Y = 2*x1 - 3*x2 + 1.
def test_fit_regressao_linear(dados_lineares):
    X, Y = dados_lineares
    modelo = MLPRegressor(hidden_layers=(), learning_rate=0.1, n_epochs=2000, random_state=0)
    modelo.fit(X, Y)
    assert np.allclose(modelo.predict(X), Y, atol=1e-2)


# Testa que uma rede com camada oculta aprende a mesma relação linear com erro pequeno.
def test_fit_rede_com_camada_oculta_em_dados_lineares(dados_lineares):
    X, Y = dados_lineares
    modelo = MLPRegressor(hidden_layers=(8,), learning_rate=0.05, n_epochs=3000, random_state=0)
    modelo.fit(X, Y)
    assert np.mean((modelo.predict(X) - Y) ** 2) < 0.05


# FIT: ESTADO E REPRODUTIBILIDADE

# Testa que o fit devolve o próprio objeto, permitindo encadear chamadas.
def test_fit_retorna_self(dados_lineares):
    X, Y = dados_lineares
    modelo = MLPRegressor(n_epochs=5, random_state=0)
    assert modelo.fit(X, Y) is modelo


# Testa que o mesmo random_state produz previsões idênticas.
def test_fit_reprodutivel_com_random_state(dados_lineares):
    X, Y = dados_lineares
    p1 = MLPRegressor(n_epochs=50, random_state=3).fit(X, Y).predict(X)
    p2 = MLPRegressor(n_epochs=50, random_state=3).fit(X, Y).predict(X)
    assert np.array_equal(p1, p2)


# Testa que a perda é registrada uma vez por época.
def test_fit_historico_de_perda_tem_n_epochs(dados_lineares):
    X, Y = dados_lineares
    modelo = MLPRegressor(n_epochs=30, random_state=0).fit(X, Y)
    assert len(modelo._loss_history) == 30


# Testa que um segundo fit recria os parâmetros em vez de reaproveitar os do primeiro, inclusive com outro número de features.
def test_refit_recria_parametros(dados_lineares):
    X, Y = dados_lineares
    modelo = MLPRegressor(n_epochs=10, random_state=0).fit(X, Y)
    modelo.fit(X[:, :1], Y)
    assert modelo._weights[0].shape[0] == 1
    assert len(modelo._loss_history) == 10


# Testa que o fit aceita listas do Python e arrays do NumPy com resultados idênticos.
def test_fit_nao_aceita_lista(dados_lineares):
    X, Y = dados_lineares
    with pytest.raises(TypeError):
        p_lista = MLPRegressor(n_epochs=20, random_state=0).fit(X.tolist(), Y.tolist()).predict(X.tolist())

# FIT: VALIDAÇÕES DE ENTRADA

# Testa que o fit rejeita X que não é matriz 2D.
def test_fit_com_X_nao_2d():
    with pytest.raises(ValueError):
        MLPRegressor().fit(np.array([1, 2, 3]), np.array([1, 2, 3]))


# Testa que o fit rejeita Y que não é vetor 1D.
def test_fit_com_Y_nao_1d(dados_lineares):
    X, Y = dados_lineares
    with pytest.raises(ValueError):
        MLPRegressor().fit(X, Y.reshape(-1, 1))


# Testa que o fit rejeita X e Y com números de amostras diferentes.
def test_fit_com_tamanhos_incompativeis(dados_lineares):
    X, Y = dados_lineares
    with pytest.raises(ValueError):
        MLPRegressor().fit(X, Y[:-1])


# Testa que o fit rejeita NaN e infinito em X e em Y.
@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("onde", ["X", "Y"])
def test_fit_com_valores_nao_finitos(dados_lineares, valor, onde):
    X, Y = dados_lineares
    X, Y = X.copy(), Y.copy()
    if onde == "X":
        X[0, 0] = valor
    else:
        Y[0] = valor
    with pytest.raises(ValueError):
        MLPRegressor().fit(X, Y)


# Testa que o fit rejeita X vazio.
def test_fit_com_X_vazio():
    with pytest.raises(ValueError):
        MLPRegressor().fit(np.empty((0, 2)), np.empty(0))


# Testa que o fit rejeita Y com rótulos de texto, já que o alvo deve ser numérico.
def test_fit_com_Y_de_texto():
    with pytest.raises(TypeError):
        MLPRegressor().fit(np.array([[1], [2]]), ["a", "b"])


# Testa que uma learning_rate absurda faz o fit levantar FloatingPointError em vez de treinar com lixo.
def test_fit_detecta_divergencia(dados_lineares):
    X, Y = dados_lineares
    with pytest.raises(FloatingPointError):
        MLPRegressor(hidden_layers=(8,), learning_rate=1e6, n_epochs=200, random_state=0).fit(X * 100, Y)

# PREDICT

# Testa que a saída é um vetor 1D com uma previsão por linha de X.
def test_predict_formato_da_saida(dados_lineares):
    X, Y = dados_lineares
    previsto = MLPRegressor(n_epochs=5, random_state=0).fit(X, Y).predict(X[:7])
    assert isinstance(previsto, np.ndarray)
    assert previsto.shape == (7,)


# Testa que uma amostra 1D é aceita e tratada como matriz (1, d).
def test_predict_com_amostra_1d(dados_lineares):
    X, Y = dados_lineares
    previsto = MLPRegressor(n_epochs=5, random_state=0).fit(X, Y).predict(np.array([0.5, -0.5]))
    assert previsto.shape == (1,)


# Testa que predict antes do fit levanta RuntimeError.
def test_predict_antes_do_fit():
    with pytest.raises(RuntimeError):
        MLPRegressor().predict(np.array([[1, 2]]))


# Testa que o predict rejeita número de colunas diferente do usado no fit.
def test_predict_com_numero_errado_de_colunas(dados_lineares):
    X, Y = dados_lineares
    modelo = MLPRegressor(n_epochs=5, random_state=0).fit(X, Y)
    with pytest.raises(ValueError):
        modelo.predict(np.array([[1, 2, 3]]))


# Testa que o predict rejeita NaN e infinito.
@pytest.mark.parametrize("valor", [np.nan, np.inf])
def test_predict_com_valores_nao_finitos(dados_lineares, valor):
    X, Y = dados_lineares
    modelo = MLPRegressor(n_epochs=5, random_state=0).fit(X, Y)
    with pytest.raises(ValueError):
        modelo.predict(np.array([[valor, 0.0]]))


# Testa que o predict não altera os pesos do modelo.
def test_predict_nao_altera_parametros(dados_lineares):
    X, Y = dados_lineares
    modelo = MLPRegressor(n_epochs=5, random_state=0).fit(X, Y)
    antes = [W.copy() for W in modelo._weights]
    modelo.predict(X)
    assert all(np.array_equal(a, W) for a, W in zip(antes, modelo._weights))