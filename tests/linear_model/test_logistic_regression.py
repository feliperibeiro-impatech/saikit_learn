import numpy as np
import pytest

from saikit_learn.linear_model import LogisticRegression


# DADOS DE TESTE

@pytest.fixture
def separable_data():
    X = np.array([
        [-2.0, -2.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    y = np.array([0, 0, 1, 1])

    return X, y


@pytest.fixture
def non_separable_data():
    X = np.array([
        [-2.0, -1.0],
        [-1.0, -1.0],
        [-1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ])

    y = np.array([0, 0, 0, 0, 1, 1, 1, 0])

    return X, y


# TESTES DO CONSTRUTOR

@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"learning_rate": -0.1},
        {"learning_rate": np.inf},
        {"learning_rate": np.nan},
        {"learning_rate": True},
        {"learning_rate": "0.1"},

        {"max_iter": 0},
        {"max_iter": -1},
        {"max_iter": True},
        {"max_iter": False},
        {"max_iter": 10.5},

        {"tol": -0.1},
        {"tol": np.inf},
        {"tol": np.nan},
        {"tol": True},
        {"tol": "0.01"},

        {"alpha": -0.1},
        {"alpha": np.inf},
        {"alpha": np.nan},
        {"alpha": True},
        {"alpha": "0.1"},
    ],
)
def test_constructor_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        LogisticRegression(**kwargs)


def test_constructor_accepts_numpy_numeric_types():
    model = LogisticRegression(
        learning_rate=np.float64(0.1),
        max_iter=np.int64(100),
        tol=np.float64(1e-6),
        alpha=np.float64(0.01),
    )

    assert model.learning_rate == 0.1
    assert model.max_iter == 100
    assert model.tol == 1e-6
    assert model.alpha == 0.01


# TESTES DA SIGMOID

def test_sigmoid_basic_values():
    model = LogisticRegression()

    z = np.array([-1.0, 0.0, 1.0])

    result = model._sigmoid(z)

    expected = np.array([
        1 / (1 + np.exp(1)),
        0.5,
        1 / (1 + np.exp(-1)),
    ])

    np.testing.assert_allclose(result, expected)


def test_sigmoid_extreme_values_are_finite():
    model = LogisticRegression()

    z = np.array([-1000.0, 0.0, 1000.0])

    result = model._sigmoid(z)

    assert np.isfinite(result).all()

    np.testing.assert_allclose(
        result,
        np.array([0.0, 0.5, 1.0]),
    )


# TESTES DA LOSS

def test_loss_is_finite_for_extreme_probabilities():
    model = LogisticRegression()

    y = np.array([0.0, 1.0])

    probabilities = np.array([0.0, 1.0])

    loss = model._loss(y, probabilities)

    assert np.isfinite(loss)


def test_loss_is_positive_for_incorrect_predictions():
    model = LogisticRegression()

    y = np.array([0.0, 1.0])

    probabilities = np.array([0.9, 0.1])

    loss = model._loss(y, probabilities)

    assert loss > 0


# TESTES DE FIT

def test_fit_returns_self(separable_data):
    X, y = separable_data

    model = LogisticRegression()

    result = model.fit(X, y)

    assert result is model


def test_fit_learns_two_classes(separable_data):
    X, y = separable_data

    model = LogisticRegression().fit(X, y)

    np.testing.assert_array_equal(
        model.classes_,
        np.array([0, 1]),
    )


def test_fit_sets_expected_attributes(separable_data):
    X, y = separable_data

    model = LogisticRegression().fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert model.n_features_in_ == X.shape[1]


def test_fit_rejects_non_numpy_X():
    X = [[1.0], [2.0]]
    y = np.array([0, 1])

    with pytest.raises(TypeError):
        LogisticRegression().fit(X, y)


def test_fit_rejects_non_numpy_y():
    X = np.array([[1.0], [2.0]])
    y = [0, 1]

    with pytest.raises(TypeError):
        LogisticRegression().fit(X, y)


def test_fit_rejects_non_2d_X():
    X = np.array([1.0, 2.0, 3.0])
    y = np.array([0, 1, 0])

    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y)


def test_fit_rejects_invalid_y_shape():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    y = np.array([
        [0],
        [1],
    ])

    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y)


def test_fit_rejects_different_number_of_samples():
    X = np.array([
        [1.0],
        [2.0],
    ])

    y = np.array([0, 1, 0])

    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y)


def test_fit_rejects_nan_in_X():
    X = np.array([
        [1.0],
        [np.nan],
    ])

    y = np.array([0, 1])

    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y)


def test_fit_rejects_inf_in_X():
    X = np.array([
        [1.0],
        [np.inf],
    ])

    y = np.array([0, 1])

    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y)


def test_fit_rejects_X_with_no_samples():
    X = np.empty((0, 2))
    y = np.array([])

    with pytest.raises(ValueError, match="X não pode ser vazio"):
        LogisticRegression().fit(X, y)


def test_fit_rejects_X_with_no_features():
    X = np.empty((2, 0))
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="X não pode ser vazio"):
        LogisticRegression().fit(X, y)


def test_fit_requires_exactly_two_classes():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
    ])

    y = np.array([0, 1, 2])

    with pytest.raises(
        ValueError,
        match="exatamente duas classes",
    ):
        LogisticRegression().fit(X, y)


def test_fit_rejects_only_one_class():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
    ])

    y = np.array([1, 1, 1])

    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y)


# TESTES DO TREINAMENTO

def test_loss_history_is_populated_and_finite(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=100,
    ).fit(X, y)

    assert len(model.loss_history) > 0
    assert np.isfinite(model.loss_history).all()


def test_loss_decreases_on_controlled_data(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=100,
    ).fit(X, y)

    assert model.loss_history[-1] < model.loss_history[0]


def test_fit_resets_loss_history(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=10,
    )

    model.fit(X, y)

    first_length = len(model.loss_history)

    model.fit(X, y)

    assert len(model.loss_history) == first_length


def test_n_iter_matches_loss_history_length(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=100,
    ).fit(X, y)

    assert model.n_iter_ == len(model.loss_history)

    assert 1 <= model.n_iter_ <= model.max_iter


def test_converged_is_true_with_regularization(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=10000,
        tol=1e-4,
        alpha=0.1,
    ).fit(X, y)

    assert model.converged_ is True
    assert model.n_iter_ < model.max_iter


def test_converged_is_false_when_max_iter_is_reached(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=1,
        tol=1e-15,
    ).fit(X, y)

    assert model.n_iter_ == 1
    assert model.converged_ is False


def test_n_iter_is_reset_on_each_fit(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=100,
    )

    model.fit(X, y)

    first_n_iter = model.n_iter_

    model.fit(X, y)

    assert model.n_iter_ == first_n_iter
    assert model.n_iter_ == len(model.loss_history)


# TESTES DA REGULARIZAÇÃO L2

def test_alpha_zero_is_equivalent_to_default(separable_data):
    X, y = separable_data

    default_model = LogisticRegression(
        learning_rate=0.1,
        max_iter=1000,
        tol=1e-9,
    ).fit(X, y)

    alpha_zero_model = LogisticRegression(
        learning_rate=0.1,
        max_iter=1000,
        tol=1e-9,
        alpha=0.0,
    ).fit(X, y)

    np.testing.assert_allclose(
        default_model.coef_,
        alpha_zero_model.coef_,
    )

    np.testing.assert_allclose(
        default_model.intercept_,
        alpha_zero_model.intercept_,
    )


def test_l2_regularization_reduces_coefficient_norm(separable_data):
    X, y = separable_data

    model_without_regularization = LogisticRegression(
        learning_rate=0.1,
        max_iter=5000,
        tol=1e-8,
        alpha=0.0,
    ).fit(X, y)

    model_with_regularization = LogisticRegression(
        learning_rate=0.1,
        max_iter=5000,
        tol=1e-8,
        alpha=0.1,
    ).fit(X, y)

    norm_without = np.linalg.norm(
        model_without_regularization.coef_
    )

    norm_with = np.linalg.norm(
        model_with_regularization.coef_
    )

    assert norm_with < norm_without


def test_l2_does_not_penalize_intercept_directly():
    X = np.array([
        [-2.0],
        [-1.0],
        [1.0],
        [2.0],
    ])

    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=1000,
        alpha=1.0,
    ).fit(X, y)

    assert np.isfinite(model.intercept_)


# TESTES DE PREDICT_PROBA

def test_predict_proba_before_fit_raises_error():
    model = LogisticRegression()

    X = np.array([[1.0, 2.0]])

    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_predict_proba_returns_correct_shape(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=1000,
    ).fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)


def test_predict_proba_rows_sum_to_one(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=1000,
    ).fit(X, y)

    probabilities = model.predict_proba(X)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X)),
    )


def test_predict_proba_accepts_single_sample(separable_data):
    X, y = separable_data

    model = LogisticRegression().fit(X, y)

    probabilities = model.predict_proba(
        np.array([0.0, 0.0])
    )

    assert probabilities.shape == (1, 2)


def test_predict_proba_rejects_wrong_number_of_features(
    separable_data,
):
    X, y = separable_data

    model = LogisticRegression().fit(X, y)

    X_wrong = np.array([
        [1.0, 2.0, 3.0],
    ])

    with pytest.raises(ValueError):
        model.predict_proba(X_wrong)


# TESTES DE PREDICT

def test_predict_before_fit_raises_error():
    model = LogisticRegression()

    X = np.array([[1.0, 2.0]])

    with pytest.raises(RuntimeError):
        model.predict(X)


def test_predict_returns_correct_classes(separable_data):
    X, y = separable_data

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=5000,
        tol=1e-6,
    ).fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_array_equal(
        predictions,
        y,
    )


def test_predict_returns_original_class_labels():
    X = np.array([
        [-2.0],
        [-1.0],
        [1.0],
        [2.0],
    ])

    y = np.array([
        "negative",
        "negative",
        "positive",
        "positive",
    ])

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=5000,
    ).fit(X, y)

    predictions = model.predict(X)

    assert set(predictions).issubset(
        set(y)
    )