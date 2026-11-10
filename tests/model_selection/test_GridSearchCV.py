import numpy as np
import pytest
from unittest.mock import patch

from saikit_learn.model_selection.GridSearchCV import GridSearchCV

class DummyEstimator:
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.fitted = False

    def fit(self, x, y):
        self.fitted = True
        return self

def dummy_scoring(y_true, y_pred):
    return 1.0

@pytest.fixture
def data():
    x = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 3, 4])
    return x, y

@pytest.fixture
def param_grid():
    return {
        "alpha": [1, 2],
        "beta": [10, 20]
    }

def test_param_grid_generation(param_grid):
    gs = GridSearchCV(
        DummyEstimator,
        param_grid,
        scoring=dummy_scoring,
        refit=False
    )

    assert len(gs.param_grid) == 4

    assert gs.param_grid == [
        {"alpha": 1, "beta": 10},
        {"alpha": 1, "beta": 20},
        {"alpha": 2, "beta": 10},
        {"alpha": 2, "beta": 20}
    ]

@patch("saikit_learn.model_selection.GridSearchCV.cross_validation")
def test_fit_returns_best_parameters(mock_cv, data, param_grid):
    x, y = data

    mock_cv.side_effect = [
        [0.7, 0.8, 0.9],
        [0.6, 0.7, 0.8],
        [0.9, 0.95, 0.85],
        [0.5, 0.6, 0.7]
    ]

    gs = GridSearchCV(
        DummyEstimator,
        param_grid,
        scoring=dummy_scoring,
        refit=False,
        cv=3
    )

    results, best_score, best_index, best_params, best_estimator = gs.fit(x, y)

    assert best_score == pytest.approx(0.9)
    assert best_index == 2
    assert best_params == {"alpha": 2, "beta": 10}

    assert isinstance(best_estimator, DummyEstimator)
    assert best_estimator.alpha == 2
    assert best_estimator.beta == 10


@patch("saikit_learn.model_selection.GridSearchCV.cross_validation")
def test_results_contain_scores_and_rankings(mock_cv, data, param_grid):
    x, y = data

    mock_cv.side_effect = [
        [0.7, 0.8],
        [0.6, 0.7],
        [0.9, 1.0],
        [0.5, 0.6]
    ]

    gs = GridSearchCV(
        DummyEstimator,
        param_grid,
        scoring=dummy_scoring,
        refit=False,
        cv=2
    )

    results, _, _, _, _ = gs.fit(x, y)

    assert results["alpha"] == [1, 1, 2, 2]
    assert results["beta"] == [10, 20, 10, 20]

    assert results["score"] == pytest.approx([
        0.75, 0.65, 0.95, 0.55
    ])

    assert list(results["ranking"]) == [2, 3, 1, 4]

@patch("saikit_learn.model_selection.GridSearchCV.cross_validation")
def test_cross_validation_called_for_each_combination(
    mock_cv, data, param_grid
):
    x, y = data
    mock_cv.return_value = [0.8, 0.9]

    gs = GridSearchCV(
        DummyEstimator,
        param_grid,
        scoring=dummy_scoring,
        refit=False,
        cv=2
    )

    gs.fit(x, y)

    assert mock_cv.call_count == 4

    for call in mock_cv.call_args_list:
        estimator, received_x, received_y, scoring, cv = call.args

        assert isinstance(estimator, DummyEstimator)
        assert np.array_equal(received_x, x)
        assert np.array_equal(received_y, y)
        assert scoring is dummy_scoring
        assert cv == 2

    estimators = [
        (call.args[0].alpha, call.args[0].beta)
        for call in mock_cv.call_args_list
    ]

    assert estimators == [
        (1, 10),
        (1, 20),
        (2, 10),
        (2, 20)
    ]

@patch("saikit_learn.model_selection.GridSearchCV.cross_validation")
def test_refit_true_returns_fitted_estimator(mock_cv, data, param_grid):
    x, y = data
    mock_cv.return_value = [0.8, 0.9]

    gs = GridSearchCV(
        DummyEstimator,
        param_grid,
        scoring=dummy_scoring,
        refit=True,
        cv=2
    )

    best_estimator = gs.fit(x, y)

    assert isinstance(best_estimator, DummyEstimator)
    assert best_estimator.fitted is True

@patch("saikit_learn.model_selection.GridSearchCV.cross_validation")
def test_refit_false_does_not_fit_best_estimator(
    mock_cv, data, param_grid
):
    x, y = data
    mock_cv.return_value = [0.8, 0.9]

    gs = GridSearchCV(
        DummyEstimator,
        param_grid,
        scoring=dummy_scoring,
        refit=False,
        cv=2
    )

    _, _, _, _, best_estimator = gs.fit(x, y)

    assert best_estimator.fitted is False

@patch("saikit_learn.model_selection.GridSearchCV.cross_validation")
def test_cv_parameter_is_passed_correctly(mock_cv, data):
    x, y = data
    mock_cv.return_value = [0.8, 0.9]

    gs = GridSearchCV(
        DummyEstimator,
        {"alpha": [1]},
        scoring=dummy_scoring,
        refit=False,
        cv=3
    )

    gs.fit(x, y)

    assert mock_cv.call_count == 1

    call_args = mock_cv.call_args.args

    estimator, received_x, received_y, scoring, cv = call_args

    assert isinstance(estimator, DummyEstimator)
    assert estimator.alpha == 1
    assert np.array_equal(received_x, x)
    assert np.array_equal(received_y, y)
    assert scoring is dummy_scoring
    assert cv == 3