import numpy as np
import pytest

from saikit_learn.linear_model import LogisticRegression


# FIXTURES

@pytest.fixture
def separable_data():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    return X, y


@pytest.fixture
def overlapping_data():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(-1.0, 1.0, size=(100, 2)),
        rng.normal(1.0, 1.0, size=(100, 2)),
    ])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


@pytest.fixture
def three_classes():
    rng = np.random.default_rng(0)
    means = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    X = np.vstack([rng.normal(mean, 1.0, size=(80, 2)) for mean in means])
    y = np.repeat([0, 1, 2], 80)
    return X, y


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    p = 1 / (1 + np.exp(-(X @ np.array([1.5, -2.0, 0.5]) + 0.3)))
    y = (rng.random(300) < p).astype(int)
    return X, y


# TREINAMENTO BINÁRIO

def test_fit_returns_self_and_learns_separable_data(separable_data):
    X, y = separable_data
    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=5000,
        tol=1e-9,
    )

    result = model.fit(X, y)

    assert result is model
    np.testing.assert_array_equal(model.predict(X), y)


def test_loss_history_is_valid(separable_data):
    X, y = separable_data
    model = LogisticRegression(learning_rate=0.1, max_iter=100).fit(X, y)

    assert len(model.loss_history) == 1
    assert 1 <= len(model.loss_history[0]) <= model.max_iter
    assert np.isfinite(model.loss_history[0]).all()
    assert model.loss_history[0][-1] < model.loss_history[0][0]


def test_predict_returns_original_labels():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array(["cat", "cat", "dog", "dog"])

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=3000,
    ).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(model.classes_, ["cat", "dog"])


# PREDIÇÃO

def test_predict_proba_returns_valid_probabilities(separable_data):
    X, y = separable_data
    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=3000,
    ).fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_single_sample_prediction(separable_data):
    X, y = separable_data
    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=3000,
    ).fit(X, y)

    probabilities = model.predict_proba(np.array([1.5]))
    prediction = model.predict(np.array([1.5]))

    assert probabilities.shape == (1, 2)
    assert prediction.shape == (1,)
    assert prediction[0] == 1


def test_predict_before_fit_raises_runtime_error():
    model = LogisticRegression()

    with pytest.raises(RuntimeError, match="Chame fit"):
        model.predict(np.array([[1.0]]))

    with pytest.raises(RuntimeError, match="Chame fit"):
        model.predict_proba(np.array([[1.0]]))


# VALIDAÇÃO

def test_fit_rejects_non_numpy_inputs():
    model = LogisticRegression()

    with pytest.raises(TypeError, match="X deve ser np.ndarray"):
        model.fit([[-1.0], [1.0]], np.array([0, 1]))

    with pytest.raises(TypeError, match="y deve ser np.ndarray"):
        model.fit(np.array([[-1.0], [1.0]]), [0, 1])


def test_fit_rejects_invalid_shapes_and_classes():
    model = LogisticRegression()

    with pytest.raises(ValueError, match="X deve ser 2D"):
        model.fit(np.array([1.0, 2.0]), np.array([0, 1]))

    with pytest.raises(ValueError, match="y deve ser 1D"):
        model.fit(
            np.array([[1.0], [2.0]]),
            np.array([[0], [1]]),
        )

    with pytest.raises(ValueError, match="pelo menos duas classes"):
        model.fit(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([0, 0, 0]),
        )


def test_rejects_invalid_X_values(separable_data):
    X, y = separable_data
    fitted = LogisticRegression().fit(X, y)

    with pytest.raises(ValueError, match="não finitos"):
        LogisticRegression().fit(
            np.array([[0.0], [np.nan]]),
            np.array([0, 1]),
        )

    with pytest.raises(ValueError, match="numéricos"):
        fitted.predict_proba(np.array([["a"]]))


def test_predict_proba_validates_features(separable_data):
    X, y = separable_data
    model = LogisticRegression().fit(X, y)

    with pytest.raises(ValueError, match="shape"):
        model.predict_proba(np.array([[1.0, 2.0]]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"learning_rate": np.nan},
        {"learning_rate": True},
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"max_iter": True},
        {"tol": -1},
        {"tol": np.inf},
        {"alpha": -0.1},
        {"alpha": np.nan},
        {"alpha": True},
    ],
)
def test_constructor_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        LogisticRegression(**kwargs)


# ESTADO E REFIT

def test_public_attributes_before_fit():
    model = LogisticRegression()

    assert model.classes_ is None
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model.n_features_in_ is None
    assert model.n_iter_ is None
    assert model.converged_ is None


def test_failed_fit_does_not_change_fitted_model(overlapping_data):
    X, y = overlapping_data
    model = LogisticRegression(learning_rate=0.5, max_iter=200).fit(X, y)

    coef_before = model.coef_.copy()
    intercept_before = model.intercept_.copy()

    with pytest.raises(ValueError, match="pelo menos duas classes"):
        model.fit(np.zeros((10, 3)), np.zeros(10))

    np.testing.assert_array_equal(model.coef_, coef_before)
    np.testing.assert_array_equal(model.intercept_, intercept_before)
    assert model.n_features_in_ == 2

    assert model.predict_proba(X).shape == (len(X), 2)


def test_refit_updates_model_shape(data):
    X, y = data
    model = LogisticRegression(learning_rate=0.5, max_iter=500).fit(X, y)

    model.fit(X[:, :2], y)

    assert model.n_features_in_ == 2
    assert model.coef_.shape == (1, 2)

    with pytest.raises(ValueError, match="shape"):
        model.predict_proba(X)


# CONVERGÊNCIA

def test_convergence_attributes(overlapping_data):
    X, y = overlapping_data

    model = LogisticRegression(
        learning_rate=0.5,
        max_iter=20000,
        tol=1e-5,
    ).fit(X, y)

    assert model.converged_.all()
    assert 1 <= model.n_iter_[0] < model.max_iter
    assert model.n_iter_[0] == len(model.loss_history[0])


def test_does_not_converge_when_max_iter_is_reached(overlapping_data):
    X, y = overlapping_data

    model = LogisticRegression(
        learning_rate=0.01,
        max_iter=5,
        tol=1e-12,
    ).fit(X, y)

    assert not model.converged_.any()
    assert model.n_iter_[0] == 5


def test_stopping_criterion_does_not_depend_on_learning_rate(overlapping_data):
    X, y = overlapping_data

    model = LogisticRegression(
        learning_rate=1e-7,
        max_iter=50,
        tol=1e-6,
    ).fit(X, y)

    assert model.n_iter_[0] == 50
    assert not model.converged_.any()


# REGULARIZAÇÃO L2

def test_alpha_zero_matches_default_model(overlapping_data):
    X, y = overlapping_data

    default = LogisticRegression(
        learning_rate=0.5,
        max_iter=500,
    ).fit(X, y)

    explicit = LogisticRegression(
        learning_rate=0.5,
        max_iter=500,
        alpha=0.0,
    ).fit(X, y)

    np.testing.assert_array_equal(default.coef_, explicit.coef_)
    np.testing.assert_array_equal(default.intercept_, explicit.intercept_)


def test_regularization_shrinks_coefficients(overlapping_data):
    X, y = overlapping_data

    without_l2 = LogisticRegression(
        learning_rate=0.5,
        max_iter=50000,
        tol=1e-9,
        alpha=0.0,
    ).fit(X, y)

    with_l2 = LogisticRegression(
        learning_rate=0.3,
        max_iter=50000,
        tol=1e-9,
        alpha=1.0,
    ).fit(X, y)

    assert np.linalg.norm(with_l2.coef_) < np.linalg.norm(without_l2.coef_)


def test_intercept_is_not_penalized():
    X = np.zeros((10, 1))
    y = np.array([1] * 8 + [0] * 2)

    model = LogisticRegression(
        learning_rate=0.01,
        max_iter=50000,
        tol=1e-9,
        alpha=100.0,
    ).fit(X, y)

    np.testing.assert_allclose(
        model.intercept_[0],
        np.log(4.0),
        atol=1e-6,
    )


# REFERÊNCIA INDEPENDENTE

def newton_reference(X, y, alpha):
    """Solução de referência por Newton-Raphson."""
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    penalty = np.diag([1.0] * d + [0.0])
    w = np.zeros(d + 1)

    for _ in range(100):
        p = 1 / (1 + np.exp(-A @ w))
        gradient = A.T @ (p - y) / n + alpha * (penalty @ w)
        hessian = (
            A.T @ (A * (p * (1 - p))[:, None]) / n
            + alpha * penalty
        )
        w -= np.linalg.solve(hessian, gradient)

    return w[:d], w[d]


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_matches_newton_raphson_solution(data, alpha):
    X, y = data

    model = LogisticRegression(
        learning_rate=min(0.5, 1.0 / (1.0 + alpha)),
        max_iter=200000,
        tol=1e-10,
        alpha=alpha,
    ).fit(X, y)

    coef, intercept = newton_reference(X, y, alpha)

    assert model.converged_.all()
    np.testing.assert_allclose(model.coef_[0], coef, atol=1e-7)
    np.testing.assert_allclose(model.intercept_[0], intercept, atol=1e-7)


# ESTABILIDADE NUMÉRICA

def test_sigmoid_extreme_values_do_not_overflow():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = LogisticRegression()._sigmoid(z)

    np.testing.assert_allclose(
        result,
        [0.0, 0.0, 0.5, 1.0, 1.0],
        atol=1e-12,
    )


# MULTICLASSE

def test_multiclass_attributes_and_predictions(three_classes):
    X, y = three_classes

    model = LogisticRegression(
        learning_rate=0.5,
        max_iter=5000,
    ).fit(X, y)

    assert model.coef_.shape == (3, 2)
    assert model.intercept_.shape == (3,)
    assert model.n_iter_.shape == (3,)
    assert model.converged_.shape == (3,)
    assert len(model.loss_history) == 3
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 3)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    expected = model.classes_[np.argmax(probabilities, axis=1)]
    np.testing.assert_array_equal(model.predict(X), expected)

    assert np.mean(model.predict(X) == y) > 0.9


def test_multiclass_string_labels(three_classes):
    X, y = three_classes
    labels = np.array(["gato", "cao", "rato"])
    y_text = labels[y]

    model = LogisticRegression(
        learning_rate=0.5,
        max_iter=5000,
    ).fit(X, y_text)

    np.testing.assert_array_equal(
        model.classes_,
        ["cao", "gato", "rato"],
    )

    predictions = model.predict(X)

    assert set(np.unique(predictions)) <= set(labels)
    assert np.mean(predictions == y_text) > 0.9


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_each_ovr_classifier_matches_newton_raphson(
    three_classes,
    alpha,
):
    X, y = three_classes

    model = LogisticRegression(
        learning_rate=min(0.5, 1.0 / (1.0 + alpha)),
        max_iter=300000,
        tol=1e-10,
        alpha=alpha,
    ).fit(X, y)

    assert model.converged_.all()

    for k in range(3):
        coef, intercept = newton_reference(
            X,
            (y == k).astype(float),
            alpha,
        )

        np.testing.assert_allclose(
            model.coef_[k],
            coef,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            model.intercept_[k],
            intercept,
            atol=1e-6,
        )


def test_multiclass_single_sample(three_classes):
    X, y = three_classes

    model = LogisticRegression(
        learning_rate=0.5,
        max_iter=5000,
    ).fit(X, y)

    probabilities = model.predict_proba(X[0])
    prediction = model.predict(X[0])

    assert probabilities.shape == (1, 3)
    assert prediction.shape == (1,)


def test_refit_between_binary_and_multiclass(three_classes):
    X, y = three_classes

    model = LogisticRegression(
        learning_rate=0.5,
        max_iter=1000,
    ).fit(X, y)

    assert model.coef_.shape == (3, 2)

    mask = y < 2
    model.fit(X[mask], y[mask])

    assert model.coef_.shape == (1, 2)
    assert model.predict_proba(X).shape == (len(X), 2)

    model.fit(X, y)

    assert model.coef_.shape == (3, 2)
    assert model.predict_proba(X).shape == (len(X), 3)


def test_multiclass_convergence_is_reported_per_classifier(three_classes):
    X, y = three_classes

    model = LogisticRegression(
        learning_rate=0.5,
        max_iter=7,
        tol=1e-12,
    ).fit(X, y)

    assert not model.converged_.any()
    np.testing.assert_array_equal(model.n_iter_, [7, 7, 7])
    assert [len(history) for history in model.loss_history] == [7, 7, 7]


def test_multiclass_probabilities_remain_finite_for_extreme_scores(three_classes):
    X, y = three_classes

    model = LogisticRegression(
        learning_rate=0.5,
        max_iter=1000,
    ).fit(X, y)

    model.coef_ = np.zeros_like(model.coef_)
    model.intercept_ = np.full(3, -1000.0)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        probabilities = model.predict_proba(X[:5])

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities, 1 / 3)