import numpy as np
import pytest

from saikit_learn.resampling.cross_validation import cross_validation
from saikit_learn.neighbors.knn import KNN
from saikit_learn.linear_model.logistic_regression import LogisticRegression
from saikit_learn.regression.polinomial_regression import PolReg
from saikit_learn.neural_network.mlp_regressor import MLPRegressor


class TestModel:
    """Modelo simples usado para testar a validação cruzada."""

    def fit(self, X, y):
        self.value_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def test_retorna_um_score_para_cada_fold():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    scores = cross_validation(TestModel(), X, y, mean_squared_error,
                              k=5, shuffle=False)

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 5


def test_funciona_quando_numero_de_amostras_nao_e_divisivel_por_k():
    X = np.arange(21).reshape(7, 3)
    y = np.arange(7)

    scores = cross_validation(TestModel(), X, y, mean_squared_error,
                              k=3, shuffle=False)

    assert len(scores) == 3
    assert np.all(np.isfinite(scores))


def test_loocv():
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6)

    scores = cross_validation(TestModel(), X, y, mean_squared_error,
                              k=len(X), shuffle=False)

    assert len(scores) == len(X)


def test_random_state_torna_resultado_reproduzivel():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    scores_1 = cross_validation(TestModel(), X, y, mean_squared_error,
                                k=5, shuffle=True, random_state=42)

    scores_2 = cross_validation(TestModel(), X, y, mean_squared_error,
                                k=5, shuffle=True, random_state=42)

    assert np.array_equal(scores_1, scores_2)


def test_X_deve_ser_numpy_array():
    X = [[1, 2], [3, 4]]
    y = np.array([0, 1])

    with pytest.raises(TypeError):
        cross_validation(TestModel(), X, y, mean_squared_error)


def test_y_deve_ser_numpy_array():
    X = np.array([[1, 2], [3, 4]])
    y = [0, 1]

    with pytest.raises(TypeError):
        cross_validation(TestModel(), X, y, mean_squared_error)


def test_X_deve_ser_2D():
    X = np.array([1, 2, 3])
    y = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        cross_validation(TestModel(), X, y, mean_squared_error)


def test_y_deve_ser_1D():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0], [1]])

    with pytest.raises(ValueError):
        cross_validation(TestModel(), X, y, mean_squared_error)


def test_X_e_y_devem_ter_mesmo_numero_de_amostras():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        cross_validation(TestModel(), X, y, mean_squared_error)


def test_dados_nao_podem_estar_vazios():
    X = np.empty((0, 2))
    y = np.array([])

    with pytest.raises(ValueError):
        cross_validation(TestModel(), X, y, mean_squared_error)


@pytest.mark.parametrize("k", [1, 0, -1])
def test_k_deve_ser_pelo_menos_2(k):
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        cross_validation(TestModel(), X, y, mean_squared_error, k=k)


def test_k_nao_pode_ser_maior_que_numero_de_amostras():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        cross_validation(TestModel(), X, y, mean_squared_error, k=4)


@pytest.mark.parametrize("k", [2.5, "5", True])
def test_k_deve_ser_inteiro(k):
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 2])

    with pytest.raises(TypeError):
        cross_validation(TestModel(), X, y, mean_squared_error, k=k)


def test_scoring_deve_ser_callable():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 2])

    with pytest.raises(TypeError):
        cross_validation(TestModel(), X, y, scoring=42)


@pytest.mark.parametrize("shuffle", [1, "True", None])
def test_shuffle_deve_ser_booleano(shuffle):
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 2])

    with pytest.raises(TypeError):
        cross_validation(TestModel(), X, y, mean_squared_error,
                         shuffle=shuffle)


@pytest.mark.parametrize("random_state", ["42", 1.5, True])
def test_random_state_deve_ser_inteiro_ou_none(random_state):
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 2])

    with pytest.raises(TypeError):
        cross_validation(TestModel(), X, y, mean_squared_error,
                         random_state=random_state)


def test_random_state_nao_pode_ser_negativo():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        cross_validation(TestModel(), X, y, mean_squared_error,
                         random_state=-1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Testes com os modelos feitos

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def test_cross_validation_com_knn():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = KNN(k=1)

    scores = cross_validation(
        model,
        X,
        y,
        accuracy,
        k=3,
        shuffle=True,
        random_state=42,
    )

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3
    assert np.all(scores == 1.0)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def test_cross_validation_com_regressao_logistica():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=10000,
    )

    scores = cross_validation(
        model,
        X,
        y,
        accuracy,
        k=3,
        shuffle=True,
        random_state=42,
    )

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)

def test_cross_validation_com_regressao_polinomial():
    X = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])

    y = X[:, 0] ** 2

    model = PolReg(d=2)

    scores = cross_validation(
        model,
        X,
        y,
        mean_squared_error,
        k=3,
        shuffle=True,
        random_state=42,
    )

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3
    assert np.all(scores < 1e-10)


def test_cross_validation_com_rede_neural():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
    ])

    y = 2 * X[:, 0] + 1

    model = MLPRegressor(
        hidden_layers=(5,),
        learning_rate=0.01,
        n_epochs=1000,
        random_state=42,
    )

    scores = cross_validation(
        model,
        X,
        y,
        mean_squared_error,
        k=3,
        shuffle=True,
        random_state=42,
    )

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)