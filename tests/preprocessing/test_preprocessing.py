import numpy as np
import pytest

from saikit_learn.preprocessing.preprocessing import StandardScaler, Pipeline
from saikit_learn.regression.ols import OLS
from saikit_learn.neighbors.knn import KNN


# DADOS DE APOIO

# Dados com colunas em escalas bem diferentes e alvo linear 
@pytest.fixture
def dados():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3)) * np.array([1.0, 50.0, 0.01]) + np.array([0.0, 100.0, 5.0])
    y = X @ np.array([2.0, 0.03, -100.0]) + 7.0 + rng.normal(scale=0.1, size=60)
    return X, y


# Transformador de teste: multiplica X por 2 e guarda o y recebido no fit
class _Dobra:

 def __init__(self):
        self.y_recebido = "fit nao chamado"

 def fit(self, X, y=None):
        self.y_recebido = y
        return self

 def transform(self, X):
        return 2 * X

 def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# Modelo de teste que prediz a soma das colunas e guarda o X visto no fit
class _ModeloSoma:

 def fit(self, X, y=None):
        self.X_fit = X
        return self

 def predict(self, X):
        return X.sum(axis=1)



# STANDARDSCALER

# Cada coluna sai com média 0 e desvio 1, e o X original fica intacto
def test_scaler_padroniza_sem_alterar_a_entrada(dados):
    X, _ = dados
    copia = X.copy()
    Z = StandardScaler().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z.std(axis=0), 1.0)
    assert np.array_equal(X, copia)

# fit retorna a instância e fit_transform = fit seguido de transform
def test_scaler_fit_retorna_self_e_fit_transform_equivale(dados):
    X, _ = dados
    scaler = StandardScaler()
    assert scaler.fit(X) is scaler
    assert np.allclose(StandardScaler().fit_transform(X), scaler.transform(X))

# No teste, transform usa média e desvio do treino, sem recalcular# 
def test_scaler_transform_usa_estatisticas_do_treino(dados):
    X, _ = dados
    Xtr, Xte = X[:40], X[40:] + 10.0
    Z = StandardScaler().fit(Xtr).transform(Xte)
    assert np.allclose(Z, (Xte - Xtr.mean(axis=0)) / Xtr.std(axis=0))
    assert not np.allclose(Z.mean(axis=0), 0.0)

# inverse_transform desfaz a padronização# 
def test_scaler_inverse_transform_recupera_X(dados):
    X, _ = dados
    scaler = StandardScaler().fit(X)
    assert np.allclose(scaler.inverse_transform(scaler.transform(X)), X)

# Coluna com desvio zero vira zeros, sem divisão por zero# 
def test_scaler_coluna_constante_nao_gera_nan(dados):
    X, _ = dados
    X = X.copy()
    X[:, 1] = 3.0
    Z = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 1], 0.0)

# with_mean=False só divide pelo desvio; with_std=False só subtrai a média
def test_scaler_opcoes_with_mean_e_with_std(dados):
    X, _ = dados
    sem_media = StandardScaler(with_mean=False).fit_transform(X)
    sem_desvio = StandardScaler(with_std=False).fit_transform(X)
    assert np.allclose(sem_media, X / X.std(axis=0))
    assert np.allclose(sem_desvio, X - X.mean(axis=0))

# PIPELINE

# Aplica os transformadores em ordem, repassa o y e o fit retorna self
def test_pipeline_ordem_dos_passos_e_repasse_do_y(dados):
    X, y = dados
    a, b, modelo = _Dobra(), _Dobra(), _ModeloSoma()
    pipe = Pipeline([("a", a), ("b", b), ("m", modelo)])
    assert pipe.fit(X, y) is pipe
    assert a.y_recebido is y and b.y_recebido is y
    assert np.allclose(modelo.X_fit, 4 * X)
    assert np.allclose(pipe.predict(X), (4 * X).sum(axis=1))


# Pipeline = padronizar com o treino e ajustar à mão; predict tem shape (n,)
def test_pipeline_com_ols_usa_so_o_treino_e_iguala_o_manual(dados):
    X, y = dados
    Xtr, ytr, Xte = X[:40], y[:40], X[40:]

    pipe = Pipeline([("sc", StandardScaler()), ("ols", OLS())]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    scaler = StandardScaler().fit(Xtr)
    manual = OLS().fit(scaler.transform(Xtr), ytr).predict(scaler.transform(Xte))

    assert pred.shape == (len(Xte),)
    assert np.allclose(pred, manual)
    assert np.allclose(pipe.steps[0][1].mean_, Xtr.mean(axis=0))

# Com um classificador no fim, os rótulos de texto são preservados
def test_pipeline_com_classificador_e_rotulos_string(dados):
    X, _ = dados
    rotulos = np.where(X[:, 0] + X[:, 1] / 50 > 2, "alto", "baixo")
    pipe = Pipeline([("sc", StandardScaler()), ("knn", KNN(k=1))]).fit(X, rotulos)
    pred = pipe.predict(X)
    assert pred.dtype.kind == "U"
    assert np.array_equal(pred, rotulos)