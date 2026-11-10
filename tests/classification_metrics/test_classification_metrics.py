import numpy as np
import pytest
from saikit_learn.classification_metrics import ClassificationMetrics

@pytest.fixture
def binary_data():
    """
    TP=2, FN=1, FP=3, TN=4  (positive class = label 1)
    confusion matrix (rows = true, cols = pred): [[4, 3], [1, 2]]
    """
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    return y_true, y_pred

@pytest.fixture
def multiclass_data():
    """
    confusion matrix (rows = true, cols = pred):
        [[2, 1, 0],
         [1, 2, 0],
         [1, 0, 3]]
    tp = [2, 2, 3], fp = [2, 1, 0], fn = [1, 1, 1]
    """
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2, 0])
    return y_true, y_pred

@pytest.fixture
def weights():
    """Weights for binary_data. Weighted matrix: [[6, 3], [1, 3]], total = 13."""
    return np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 3], dtype=float)

class TestInit:
    def test_default_labels_are_sorted_unique_values(self):
        m = ClassificationMetrics(np.array([2, 0, 1, 0]), np.array([1, 1, 2, 0]))
        assert list(m.labels) == [0, 1, 2]

    def test_default_labels_include_classes_only_present_in_y_pred(self):
        m = ClassificationMetrics(np.array([0, 0, 1]), np.array([0, 2, 1]))
        assert list(m.labels) == [0, 1, 2]
        assert m.confusion_matrix().shape == (3, 3)

    def test_string_labels(self):
        y_true = np.array(["cat", "dog", "cat", "dog", "cat"])
        y_pred = np.array(["cat", "dog", "dog", "dog", "cat"])
        m = ClassificationMetrics(y_true, y_pred)
        assert list(m.labels) == ["cat", "dog"]
        np.testing.assert_array_equal(m.confusion_matrix(), [[2, 1], [0, 2]])

    def test_custom_labels_define_class_order(self):
        m = ClassificationMetrics(np.array([0, 1]), np.array([0, 1]), labels=[1, 0])
        assert list(m.labels) == [1, 0]
        # values are encoded as indices into `labels`
        np.testing.assert_array_equal(m.y_true, [1, 0])
        np.testing.assert_array_equal(m.y_pred, [1, 0])

    def test_default_sample_weights_are_ones(self, binary_data):
        m = ClassificationMetrics(*binary_data)
        np.testing.assert_array_equal(m.sample_weights, np.ones(10))

    def test_custom_sample_weights_are_stored(self, binary_data, weights):
        m = ClassificationMetrics(*binary_data, sample_weights=weights)
        np.testing.assert_array_equal(m.sample_weights, weights)

    def test_y_true_and_y_pred_must_be_1d(self):
        y = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="1D"):
            ClassificationMetrics(y, y)

    def test_y_true_and_y_pred_must_have_same_shape(self):
        with pytest.raises(ValueError, match="mesmo formato"):
            ClassificationMetrics(np.array([0, 1, 1]), np.array([0, 1, 1, 0]))

    def test_sample_weights_must_match_shape(self):
        with pytest.raises(ValueError, match="mesmo formato"):
            ClassificationMetrics(
                np.array([0, 1, 1]), np.array([0, 1, 0]), sample_weights=np.ones(4)
            )

    def test_sample_weights_cannot_be_negative(self):
        with pytest.raises(ValueError, match="negativos"):
            ClassificationMetrics(
                np.array([0, 1, 1]),
                np.array([0, 1, 0]),
                sample_weights=np.array([1.0, -1.0, 1.0]),
            )

    def test_value_missing_from_labels_raises_value_error(self):
        with pytest.raises(ValueError, match="ausentes em labels"):
            ClassificationMetrics(
                np.array([0, 1, 2]), np.array([0, 1, 2]), labels=[0, 1]
            )

    def test_value_missing_from_labels_only_in_y_pred(self):
        with pytest.raises(ValueError, match=r"\[3\]"):
            ClassificationMetrics(
                np.array([0, 1]), np.array([0, 3]), labels=[0, 1]
            )

class TestConfusionMatrix:
    def test_binary_values_rows_are_true_cols_are_pred(self, binary_data):
        m = ClassificationMetrics(*binary_data)
        np.testing.assert_array_equal(m.confusion_matrix(), [[4, 3], [1, 2]])

    def test_multiclass_values(self, multiclass_data):
        m = ClassificationMetrics(*multiclass_data)
        np.testing.assert_array_equal(
            m.confusion_matrix(), [[2, 1, 0], [1, 2, 0], [1, 0, 3]]
        )

    def test_returns_float_matrix_of_expected_shape(self, multiclass_data):
        cm = ClassificationMetrics(*multiclass_data).confusion_matrix()
        assert cm.shape == (3, 3)
        assert cm.dtype == float

    def test_total_equals_number_of_samples(self, multiclass_data):
        cm = ClassificationMetrics(*multiclass_data).confusion_matrix()
        assert cm.sum() == 10

    def test_normalize_lines_rows_sum_to_one(self, binary_data):
        cm = ClassificationMetrics(*binary_data).confusion_matrix(normalize="lines")
        np.testing.assert_allclose(cm, [[4 / 7, 3 / 7], [1 / 3, 2 / 3]])
        np.testing.assert_allclose(cm.sum(axis=1), [1, 1])

    def test_normalize_columns_columns_sum_to_one(self, binary_data):
        cm = ClassificationMetrics(*binary_data).confusion_matrix(normalize="columns")
        np.testing.assert_allclose(cm, [[0.8, 0.6], [0.2, 0.4]])
        np.testing.assert_allclose(cm.sum(axis=0), [1, 1])

    def test_normalize_all_matrix_sums_to_one(self, binary_data):
        cm = ClassificationMetrics(*binary_data).confusion_matrix(normalize="all")
        np.testing.assert_allclose(cm, [[0.4, 0.3], [0.1, 0.2]])
        assert cm.sum() == pytest.approx(1.0)

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("normalize", ["lines", "columns", "all"])
    def test_normalization_with_empty_class_has_no_nan_or_warning(self, normalize):
        # class 2 exists in `labels` but never appears in y_true / y_pred
        m = ClassificationMetrics(
            np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), labels=[0, 1, 2]
        )
        cm = m.confusion_matrix(normalize=normalize)
        assert not np.isnan(cm).any()
        np.testing.assert_array_equal(cm[2], [0, 0, 0])

    def test_sample_weights_are_accumulated(self, binary_data, weights):
        m = ClassificationMetrics(*binary_data, sample_weights=weights)
        np.testing.assert_array_equal(m.confusion_matrix(), [[6, 3], [1, 3]])

    def test_custom_label_order_permutes_matrix(self, binary_data):
        m = ClassificationMetrics(*binary_data, labels=[1, 0])
        np.testing.assert_array_equal(m.confusion_matrix(), [[2, 1], [3, 4]])

    def test_perfect_predictions_give_diagonal_matrix(self):
        y = np.array([0, 1, 2, 2, 1, 0])
        cm = ClassificationMetrics(y, y).confusion_matrix()
        np.testing.assert_array_equal(cm, np.diag([2, 2, 2]))

class TestScoreBinary:
    def test_accuracy(self, binary_data):
        assert ClassificationMetrics(*binary_data).score("accuracy") == pytest.approx(0.6)

    def test_accuracy_not_normalized_returns_weighted_count(self, binary_data):
        m = ClassificationMetrics(*binary_data)
        assert m.score("accuracy", normalize=False) == pytest.approx(6.0)

    @pytest.mark.parametrize(
        "measure, expected",
        [("precision", 2 / 5), ("recall", 2 / 3), ("f1", 0.5)],
    )
    def test_default_positive_class_is_index_one(self, binary_data, measure, expected):
        m = ClassificationMetrics(*binary_data)
        assert m.score(measure, average="binary") == pytest.approx(expected)

    @pytest.mark.parametrize(
        "measure, expected",
        [("precision", 4 / 5), ("recall", 4 / 7), ("f1", 2 / 3)],
    )
    def test_pos_label_zero(self, binary_data, measure, expected):
        m = ClassificationMetrics(*binary_data)
        assert m.score(measure, average="binary", pos_label=0) == pytest.approx(expected)

    def test_binary_average_requires_exactly_two_classes(self, multiclass_data):
        m = ClassificationMetrics(*multiclass_data)
        with pytest.raises(ValueError, match="exige duas classes"):
            m.score("precision", average="binary")

    @pytest.mark.parametrize("measure", ["precision", "recall", "f1"])
    def test_binary_is_default_average(self, binary_data, measure):
        m = ClassificationMetrics(*binary_data)
        assert m.score(measure) == pytest.approx(m.score(measure, average="binary"))

    @pytest.mark.parametrize("measure", ["accuracy", "precision", "recall", "f1"])
    def test_perfect_predictions_score_one(self, measure):
        y = np.array([0, 1, 1, 0, 1])
        assert ClassificationMetrics(y, y).score(measure) == pytest.approx(1.0)

    @pytest.mark.parametrize("measure", ["accuracy", "precision", "recall", "f1"])
    def test_completely_wrong_predictions_score_zero(self, measure):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        assert ClassificationMetrics(y_true, y_pred).score(measure) == pytest.approx(0.0)

    def test_f1_is_harmonic_mean_of_precision_and_recall(self, binary_data):
        m = ClassificationMetrics(*binary_data)
        p = m.score("precision")
        r = m.score("recall")
        assert m.score("f1") == pytest.approx(2 * p * r / (p + r))

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("average", ["binary", "micro", "macro"])
    @pytest.mark.parametrize("measure", ["precision", "recall", "f1"])
    def test_no_true_positives_scores_zero_without_warning(self, measure, average):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        m = ClassificationMetrics(y_true, y_pred)
        assert m.score(measure, average=average) == 0

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("measure", ["precision", "recall", "f1"])
    def test_positive_class_never_predicted_scores_zero(self, measure):
        m = ClassificationMetrics(np.array([1, 1, 0, 0]), np.array([0, 0, 0, 0]),
                                  labels=[0, 1])
        assert m.score(measure, average="binary", pos_label=1) == 0

    @pytest.mark.parametrize("pos_label", ["a", 1.5, None])
    def test_pos_label_must_be_an_integer(self, binary_data, pos_label):
        m = ClassificationMetrics(*binary_data)
        with pytest.raises(TypeError):
            m.score("precision", average="binary", pos_label=pos_label)

    def test_pos_label_accepts_numpy_integers(self, binary_data):
        m = ClassificationMetrics(*binary_data)
        assert m.score("precision", pos_label=np.int64(0)) == pytest.approx(4 / 5)

class TestScoreInvalidArguments:
    @pytest.mark.parametrize("average", ["binary", "micro", "macro"])
    def test_unknown_measure_raises_value_error(self, binary_data, average):
        m = ClassificationMetrics(*binary_data)
        with pytest.raises(ValueError):
            m.score("not_a_measure", average=average)

    @pytest.mark.parametrize("average", ["weighted", "samples", None])
    def test_unsupported_average_raises_value_error(self, binary_data, average):
        m = ClassificationMetrics(*binary_data)
        with pytest.raises(ValueError):
            m.score("precision", average=average)

    def test_accuracy_ignores_average(self, binary_data):
        m = ClassificationMetrics(*binary_data)
        assert m.score("accuracy", average="weighted") == pytest.approx(0.6)

class TestScoreAveraging:
    @pytest.mark.parametrize("measure", ["precision", "recall", "f1"])
    def test_micro_on_multiclass_data(self, multiclass_data, measure):
        m = ClassificationMetrics(*multiclass_data)
        assert m.score(measure, average="micro") == pytest.approx(0.7)

    @pytest.mark.parametrize(
        "measure, expected",
        [
            ("precision", (0.5 + 2 / 3 + 1) / 3),
            ("recall", (2 / 3 + 2 / 3 + 3 / 4) / 3),
            ("f1", (4 / 7 + 2 / 3 + 6 / 7) / 3),
        ],
    )
    def test_macro_on_multiclass_data(self, multiclass_data, measure, expected):
        m = ClassificationMetrics(*multiclass_data)
        assert m.score(measure, average="macro") == pytest.approx(expected)

    def test_multiclass_accuracy(self, multiclass_data):
        m = ClassificationMetrics(*multiclass_data)
        assert m.score("accuracy") == pytest.approx(0.7)

    @pytest.mark.parametrize("average", ["micro", "macro"])
    @pytest.mark.parametrize("measure", ["precision", "recall", "f1"])
    def test_perfect_multiclass_predictions_score_one(self, measure, average):
        y = np.array([0, 1, 2, 2, 1, 0, 2])
        assert ClassificationMetrics(y, y).score(measure, average=average) == pytest.approx(1.0)

    @pytest.mark.filterwarnings("error")
    def test_macro_treats_zero_division_as_zero_without_warning(self):
        # class 2 is never predicted -> precision of that class is 0/0 -> 0
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1])
        m = ClassificationMetrics(y_true, y_pred)
        # precision per class: [1, 1/3, 0]; recall per class: [1, 1, 0]
        assert m.score("precision", average="macro") == pytest.approx(4 / 9)
        assert m.score("recall", average="macro") == pytest.approx(2 / 3)
        # f1 per class: [1, 1/2, 0]
        assert m.score("f1", average="macro") == pytest.approx(0.5)

class TestSampleWeights:
    def test_weighted_accuracy(self, binary_data, weights):
        m = ClassificationMetrics(*binary_data, sample_weights=weights)
        assert m.score("accuracy") == pytest.approx(9 / 13)
        assert m.score("accuracy", normalize=False) == pytest.approx(9.0)

    @pytest.mark.parametrize(
        "measure, expected",
        [("precision", 0.5), ("recall", 3 / 4), ("f1", 0.6)],
    )
    def test_weighted_binary_scores(self, binary_data, weights, measure, expected):
        m = ClassificationMetrics(*binary_data, sample_weights=weights)
        assert m.score(measure, average="binary") == pytest.approx(expected)

    def test_scaling_all_weights_does_not_change_normalized_scores(
        self, binary_data, weights
    ):
        a = ClassificationMetrics(*binary_data, sample_weights=weights)
        b = ClassificationMetrics(*binary_data, sample_weights=weights * 5)
        assert b.score("accuracy") == pytest.approx(a.score("accuracy"))
        for measure in ("precision", "recall", "f1"):
            assert b.score(measure) == pytest.approx(a.score(measure))
        np.testing.assert_allclose(b.confusion_matrix(), a.confusion_matrix() * 5)

    def test_zero_weight_is_equivalent_to_dropping_the_sample(self, binary_data):
        y_true, y_pred = binary_data
        w = np.ones(10)
        w[3] = 0.0
        keep = np.arange(10) != 3
        with_zero = ClassificationMetrics(y_true, y_pred, labels=[0, 1], sample_weights=w)
        dropped = ClassificationMetrics(y_true[keep], y_pred[keep], labels=[0, 1])
        np.testing.assert_array_equal(
            with_zero.confusion_matrix(), dropped.confusion_matrix()
        )
        assert with_zero.score("accuracy") == pytest.approx(dropped.score("accuracy"))
        for measure in ("precision", "recall", "f1"):
            assert with_zero.score(measure, average="macro") == pytest.approx(
                dropped.score(measure, average="macro")
            )

class TestKnownIssues:
    @pytest.mark.xfail(
        reason="loss() uses the encoded class indices (0/1) in y_pred as "
        "probabilities, so log(0) makes the result nan/inf for hard labels",
        strict=False,
    )
    def test_loss_of_perfect_predictions_is_zero(self):
        y = np.array([0, 1, 1, 0])
        assert ClassificationMetrics(y, y).loss() == pytest.approx(0.0)