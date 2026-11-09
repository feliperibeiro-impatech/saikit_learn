import numpy as np
import pytest

from saikit_learn.unsupervised.pca import PCA  # ajuste para o nome do seu módulo


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def X(rng):
    """100 amostras, 5 atributos correlacionados."""
    A = rng.normal(size=(5, 5))
    return rng.normal(size=(100, 5)) @ A


@pytest.fixture
def X_eixos(rng):
    """Variância dominante no eixo 0: primeira componente deve ser ~ [±1, 0]."""
    return rng.normal(size=(500, 2)) * np.array([10.0, 1.0])


# --------------------------------------------------------------------------
# __init__
# --------------------------------------------------------------------------
def test_init_valores_padrao():
    pca = PCA()
    assert pca.n_components is None
    assert pca.proporcao == 0.95


def test_init_atributos_aprendidos_comecam_none():
    pca = PCA()
    for attr in ["mean_", "n_features_inicial_", "n_components_", "components_",
                 "eigenvalues_", "explained_variance_", "explained_variance_ratio_"]:
        assert getattr(pca, attr) is None


# --------------------------------------------------------------------------
# _valide_matrix
# --------------------------------------------------------------------------
def test_valide_matrix_converte_lista_para_float():
    out = PCA._valide_matrix([[1, 2], [3, 4]])
    assert isinstance(out, np.ndarray)
    assert out.dtype == float
    assert out.shape == (2, 2)


@pytest.mark.parametrize("entrada", [[1, 2, 3], np.zeros((2, 2, 2)), 5.0])
def test_valide_matrix_rejeita_nao_2d(entrada):
    with pytest.raises(ValueError, match="bidimensional"):
        PCA._valide_matrix(entrada)


@pytest.mark.parametrize("entrada", [np.empty((0, 3)), np.empty((3, 0))])
def test_valide_matrix_rejeita_vazia(entrada):
    with pytest.raises(ValueError, match="pelo menos uma linha"):
        PCA._valide_matrix(entrada)


@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
def test_valide_matrix_rejeita_nao_finitos(valor):
    with pytest.raises(ValueError, match="NaN"):
        PCA._valide_matrix([[1.0, valor], [3.0, 4.0]])


def test_valide_matrix_rejeita_strings_nao_numericas():
    with pytest.raises(ValueError):
        PCA._valide_matrix([["a", "b"], ["c", "d"]])


# --------------------------------------------------------------------------
# _verify_PCA
# --------------------------------------------------------------------------
def test_verify_pca_antes_do_fit_levanta_erro():
    with pytest.raises(ValueError, match="ainda não foi ajustado"):
        PCA()._verify_PCA()


def test_verify_pca_depois_do_fit_nao_levanta(X):
    pca = PCA().fit(X)
    pca._verify_PCA()  # não deve levantar


# --------------------------------------------------------------------------
# resolve_d
# --------------------------------------------------------------------------
def test_resolve_d_antes_do_fit_levanta_erro():
    with pytest.raises(ValueError, match="ainda não foi calculada"):
        PCA().resolve_d()


@pytest.mark.parametrize("p", [0, -0.1, 1.5])
def test_resolve_d_proporcao_invalida(p):
    with pytest.raises(ValueError, match="intervalo"):
        PCA(proporcao=p).resolve_d()


@pytest.mark.parametrize("p", [0.5, 0.8, 0.95, 0.99])
def test_resolve_d_menor_d_que_atinge_proporcao(X, p):
    pca = PCA(proporcao=p).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d = pca.resolve_d()
    assert cum[d - 1] >= p
    if d > 1:
        assert cum[d - 2] < p


def test_resolve_d_retorna_int_python(X):
    assert isinstance(PCA().fit(X).resolve_d(), int)


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------
def test_fit_retorna_self(X):
    pca = PCA(n_components=2)
    assert pca.fit(X) is pca


def test_fit_shapes_dos_atributos(X):
    N, D = X.shape
    d = 3
    pca = PCA(n_components=d).fit(X)
    assert pca.n_features_inicial_ == D
    assert pca.n_components_ == d
    assert pca.mean_.shape == (D,)
    assert pca.eigenvalues_.shape == (D,)
    assert pca.explained_variance_ratio_.shape == (D,)
    assert pca.explained_variance_.shape == (d,)
    assert pca.components_.shape == (D, d)


def test_fit_media_correta(X):
    pca = PCA(n_components=2).fit(X)
    np.testing.assert_allclose(pca.mean_, X.mean(axis=0))


def test_fit_autovalores_batem_com_covariancia(X):
    pca = PCA().fit(X)
    esperado = np.sort(np.linalg.eigvalsh(np.cov(X.T, bias=True)))[::-1]
    np.testing.assert_allclose(pca.eigenvalues_, esperado, rtol=1e-8)


def test_fit_autovalores_em_ordem_decrescente(X):
    pca = PCA().fit(X)
    assert np.all(np.diff(pca.eigenvalues_) <= 1e-12)


def test_fit_razao_de_variancia_soma_um(X):
    pca = PCA().fit(X)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0)
    assert np.all(pca.explained_variance_ratio_ >= -1e-12)


def test_fit_componentes_ortonormais(X):
    pca = PCA(n_components=4).fit(X)
    np.testing.assert_allclose(pca.components_.T @ pca.components_,
                               np.eye(4), atol=1e-10)


def test_fit_primeira_componente_alinhada_ao_eixo_de_maior_variancia(X_eixos):
    pca = PCA(n_components=1).fit(X_eixos)
    np.testing.assert_allclose(np.abs(pca.components_[:, 0]), [1, 0], atol=0.05)


def test_fit_n_components_none_usa_proporcao(X):
    pca = PCA(n_components=None, proporcao=0.9).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    esperado = int(np.searchsorted(cum, 0.9) + 1)
    assert pca.n_components_ == esperado


def test_fit_proporcao_um_nao_estoura_numero_de_atributos(X):
    # erro numérico pode fazer cumsum terminar em 0.9999999...
    pca = PCA(proporcao=1.0).fit(X)
    assert 1 <= pca.n_components_ <= X.shape[1]


def test_fit_aceita_n_components_numpy_int(X):
    pca = PCA(n_components=np.int64(2)).fit(X)
    assert pca.n_components_ == 2


def test_fit_n_components_igual_a_D(X):
    pca = PCA(n_components=X.shape[1]).fit(X)
    assert pca.components_.shape == (X.shape[1], X.shape[1])


@pytest.mark.parametrize("n", [2.5, "3", True, np.True_, [2]])
def test_fit_n_components_tipo_invalido(X, n):
    with pytest.raises(ValueError, match="inteiro"):
        PCA(n_components=n).fit(X)


@pytest.mark.parametrize("n", [0, -1, 6])  # X tem 5 atributos
def test_fit_n_components_fora_do_intervalo(X, n):
    with pytest.raises(ValueError, match="entre 1 e"):
        PCA(n_components=n).fit(X)


def test_fit_variancia_total_zero():
    X_const = np.ones((10, 3))
    with pytest.raises(ValueError, match="variância total"):
        PCA().fit(X_const)


def test_fit_uma_unica_amostra_tem_variancia_zero():
    with pytest.raises(ValueError, match="variância total"):
        PCA().fit(np.array([[1.0, 2.0, 3.0]]))


@pytest.mark.parametrize("entrada", [[1, 2, 3], [[1.0, np.nan], [2.0, 3.0]]])
def test_fit_rejeita_entrada_invalida(entrada):
    with pytest.raises(ValueError):
        PCA().fit(entrada)


def test_fit_pode_ser_repetido_com_outros_dados(X, rng):
    pca = PCA(n_components=2).fit(X)
    Y = rng.normal(size=(50, 3))
    pca.fit(Y)
    assert pca.n_features_inicial_ == 3
    assert pca.components_.shape == (3, 2)


def test_fit_invariante_a_translacao(X):
    a = PCA().fit(X)
    b = PCA().fit(X + 100.0)
    np.testing.assert_allclose(a.eigenvalues_, b.eigenvalues_, rtol=1e-6)
    np.testing.assert_allclose(np.abs(a.components_), np.abs(b.components_),
                               atol=1e-6)


def test_fit_escala_multiplica_autovalores_por_c_ao_quadrado(X):
    c = 3.0
    a = PCA().fit(X)
    b = PCA().fit(c * X)
    np.testing.assert_allclose(b.eigenvalues_, c**2 * a.eigenvalues_, rtol=1e-8)
    np.testing.assert_allclose(b.explained_variance_ratio_,
                               a.explained_variance_ratio_, rtol=1e-8)


# --------------------------------------------------------------------------
# transform
# --------------------------------------------------------------------------
def test_transform_antes_do_fit_levanta_erro(X):
    with pytest.raises(ValueError, match="ainda não foi ajustado"):
        PCA().transform(X)


def test_transform_shape(X):
    Z = PCA(n_components=3).fit(X).transform(X)
    assert Z.shape == (X.shape[0], 3)


def test_transform_dados_projetados_tem_media_zero(X):
    Z = PCA(n_components=3).fit(X).transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-10)


def test_transform_variancia_das_projecoes_igual_aos_autovalores(X):
    pca = PCA(n_components=3).fit(X)
    Z = pca.transform(X)
    np.testing.assert_allclose(Z.var(axis=0), pca.explained_variance_, rtol=1e-8)


def test_transform_projecoes_descorrelacionadas(X):
    Z = PCA(n_components=3).fit(X).transform(X)
    C = np.cov(Z.T, bias=True)
    np.testing.assert_allclose(C - np.diag(np.diag(C)), 0, atol=1e-8)


def test_transform_aceita_lista(X):
    pca = PCA(n_components=2).fit(X)
    np.testing.assert_allclose(pca.transform(X.tolist()), pca.transform(X))


def test_transform_numero_de_colunas_errado(X):
    pca = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError, match="colunas"):
        pca.transform(X[:, :3])


def test_transform_rejeita_nan(X):
    pca = PCA(n_components=2).fit(X)
    X_ruim = X.copy()
    X_ruim[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pca.transform(X_ruim)


def test_transform_novos_dados_usa_media_do_treino(X, rng):
    pca = PCA(n_components=2).fit(X)
    novo = rng.normal(size=(7, X.shape[1]))
    esperado = (novo - X.mean(axis=0)) @ pca.components_
    np.testing.assert_allclose(pca.transform(novo), esperado)


# --------------------------------------------------------------------------
# fit_transform
# --------------------------------------------------------------------------
def test_fit_transform_equivale_a_fit_e_transform(X):
    a = PCA(n_components=3).fit_transform(X)
    pca = PCA(n_components=3)
    b = pca.fit(X).transform(X)
    np.testing.assert_allclose(a, b)


# --------------------------------------------------------------------------
# inverse_transform
# --------------------------------------------------------------------------
def test_inverse_transform_antes_do_fit_levanta_erro():
    with pytest.raises(ValueError, match="ainda não foi ajustado"):
        PCA().inverse_transform(np.zeros((3, 2)))


def test_inverse_transform_shape(X):
    pca = PCA(n_components=2).fit(X)
    Xr = pca.inverse_transform(pca.transform(X))
    assert Xr.shape == X.shape


def test_inverse_transform_reconstrucao_exata_com_todos_componentes(X):
    pca = PCA(n_components=X.shape[1]).fit(X)
    Xr = pca.inverse_transform(pca.transform(X))
    np.testing.assert_allclose(Xr, X, atol=1e-10)


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_inverse_transform_erro_igual_aos_autovalores_descartados(X, d):
    pca = PCA(n_components=d).fit(X)
    Xr = pca.inverse_transform(pca.transform(X))
    erro = np.mean(np.sum((X - Xr) ** 2, axis=1))
    assert erro == pytest.approx(pca.eigenvalues_[d:].sum(), rel=1e-8)


def test_inverse_transform_projecao_e_idempotente(X):
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    np.testing.assert_allclose(pca.transform(pca.inverse_transform(Z)), Z,
                               atol=1e-10)


def test_inverse_transform_numero_de_colunas_errado(X):
    pca = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError, match="colunas"):
        pca.inverse_transform(np.zeros((4, 3)))