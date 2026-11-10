import numpy as np
import pytest
from saikit_learn.resampling import bootstrap

@pytest.fixture(autouse=True)
def seed_global_rng():
    """bootstrap() draws from NumPy's global RNG, so seed it for every test."""
    np.random.seed(0)

class TestShape:
    def test_returns_numpy_array(self):
        assert isinstance(bootstrap(np.arange(5), 3), np.ndarray)

    def test_default_sample_size_is_size_of_data(self):
        assert bootstrap(np.arange(7), 4).shape == (4, 7)

    @pytest.mark.parametrize(
        "n_resamples, sample_size, expected_shape",
        [
            (3, 2, (3, 2)),
            (1, 5, (1, 5)),
            (10, 1, (10, 1)),
            (2, 12, (2, 12)),  # larger than the data: allowed (sampling with replacement)
        ],
    )
    def test_explicit_sample_size(self, n_resamples, sample_size, expected_shape):
        assert bootstrap(np.arange(5), n_resamples, sample_size).shape == expected_shape

    def test_one_resample_is_the_minimum(self):
        assert bootstrap(np.arange(5), 1).shape == (1, 5)

    def test_accepts_numpy_integers_for_n_resamples_and_sample_size(self):
        out = bootstrap(np.arange(5), np.int64(3), np.int64(2))
        assert out.shape == (3, 2)

    def test_accepts_python_list(self):
        out = bootstrap([1, 2, 3], 4)
        assert isinstance(out, np.ndarray)
        assert out.shape == (4, 3)

    def test_empty_data_with_default_sample_size_gives_empty_rows(self):
        assert bootstrap(np.array([]), 3).shape == (3, 0)

class TestValues:
    def test_all_values_come_from_data(self):
        data = np.array([10, 20, 30, 40, 50])
        out = bootstrap(data, 100)
        assert np.isin(out, data).all()

    @pytest.mark.parametrize(
        "data",
        [
            np.array([1, 2, 3]),
            np.array([1.5, 2.5, 3.5]),
            np.array(["a", "b", "c"]),
        ],
        ids=["int", "float", "str"],
    )
    def test_dtype_is_preserved(self, data):
        assert bootstrap(data, 5).dtype == data.dtype

    def test_single_element_data_repeats_that_element(self):
        out = bootstrap(np.array([7]), 4, 6)
        np.testing.assert_array_equal(out, np.full((4, 6), 7))

    def test_sampling_is_with_replacement(self):
        # Drawing 50 values from 50 without replacement would give 50 unique
        # values in every row; with replacement that is astronomically unlikely.
        out = bootstrap(np.arange(50), 20)
        assert all(len(np.unique(row)) < 50 for row in out)

    def test_resamples_are_not_all_identical(self):
        out = bootstrap(np.arange(100), 5)
        assert len({tuple(row) for row in out}) == 5

class TestInvalidInput:
    @pytest.mark.parametrize("n_resamples", [0, -1, -100])
    def test_non_positive_n_resamples_raises_value_error(self, n_resamples):
        with pytest.raises(ValueError, match="positivo"):
            bootstrap(np.arange(5), n_resamples)

    @pytest.mark.parametrize("n_resamples", [2.5, 3.0, None, "3"])
    def test_non_integer_n_resamples_raises_type_error(self, n_resamples):
        with pytest.raises(TypeError, match="Número de reamostras"):
            bootstrap(np.arange(5), n_resamples)

    @pytest.mark.parametrize("sample_size", [2.5, 2.0, None, "2"])
    def test_non_integer_sample_size_raises_type_error(self, sample_size):
        with pytest.raises(TypeError, match="Tamanho da amostra"):
            bootstrap(np.arange(5), 2, sample_size)

    def test_type_of_n_resamples_is_checked_before_its_value(self):
        # 0.5 is both the wrong type and non-positive: the type error wins
        with pytest.raises(TypeError):
            bootstrap(np.arange(5), 0.5)

    def test_empty_data_with_explicit_sample_size_raises_value_error(self):
        with pytest.raises(ValueError):
            bootstrap(np.array([]), 3, 5)