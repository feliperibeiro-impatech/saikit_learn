import numpy as np
import pytest 

from regression import MSE

def test_MSE_0_perfect_prediction():
    """
    Test the MSE function with perfect predictions.
    The MSE should be 0 when the predicted values are exactly equal to the true values
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert MSE(X, y_pred) == 0.0

def test_MSE_1_sp_case():
    """
    Test the MSE function with a simple case.
    The MSE should be 1
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert MSE(X, y_pred) == 1.0

def test_MSE_2_sp_case():
    """
    Test the MSE function with a simple case.
    The MSE should be 2/3
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 2.0])
    assert MSE(X, y_pred) == pytest.approx(2/3)

def test_MSE_3_negative_case():
    """
    Test the MSE function with a simple negative case.
    The MSE should be 5/3
    """
    X = np.array([-2.0, -1.0, 0.0])
    y_pred = np.array([-1.0, 1.0, 0.0])
    assert MSE(X, y_pred) == pytest.approx(5/3)

def test_MSE_4_decimal_case():
    """
    Test the MSE function with a simple decimal case.
    The MSE should be 1/4
    """
    X = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 1.5])
    assert MSE(X, y_pred) == pytest.approx(1/4)

def test_MSE_5_large_numbers_case():
    """
    Test the MSE function with large numbers.
    The MSE should be 6400.0, which is the square of the difference (80) between the true and predicted values.
    """
    X = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert MSE(X, y_pred) == pytest.approx(6400.0) 

def test_MSE_6_simetric_case():
    """
    Test the estabished MSE function with a simple symmetric case.
    This should be the same
    """
    X = np.array([-1.0, 0.0, 1.0])
    y_pred = np.array([1.0, 0.0, -1.0])
    assert MSE(X, y_pred) == MSE(y_pred, X)

def test_MSE_7_MSE_return():
    """
    Test the return of the MSE function
    The MSE return value should be a float
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert isinstance(MSE(X, y_pred), float)

def test_MSE_8_constant_case():
    """
    Test the MSE function with a constant prediction.
    The MSE should be equal to the original result multiplied by the constant squared 
    """
    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 5.0, 2.0])
    c = 4.0
    assert MSE((X * c),(y_pred * c)) == MSE(X, y_pred) * c * c

# def test_MSE_8_empty_case():
#     """
#     Test the MSE function with empty arrays.
#     The MSE should raise a ValueError when the input arrays are empty.
#     """
#     X = np.array([])
#     y_pred = np.array([])
#     with pytest.raises(ValueError):
#         MSE(X, y_pred)

# def test_MSE_9_different_length_case():
#     """
#     Test the MSE function with arrays of different lengths.
#     The MSE should raise a ValueError when the input arrays have different lengths.
#     """
#     X = np.array([1.0, 2.0])
#     y_pred = np.array([1.0, 2.0, 3.0])
#     with pytest.raises(ValueError):
#         MSE(X, y_pred)

# def test_MSE_10_non_numeric_case():
#     """
#     Test the MSE function with non-numeric input.
#     The MSE should raise a TypeError when the input arrays contain non-numeric values.
#     """
#     X = np.array([1.0, 2.0, 'a'])
#     y_pred = np.array([1.0, 2.0, 3.0])
#     with pytest.raises(TypeError):
#         MSE(X, y_pred)

# def test_MSE_11_nan_case():
#     """
#     Test the MSE function with NaN values.
#     The MSE should raise a ValueError when the input arrays contain NaN values.
#     """
#     X = np.array([1.0, 2.0, np.nan])
#     y_pred = np.array([1.0, 2.0, 3.0])
#     with pytest.raises(ValueError):
#         MSE(X, y_pred)

# def test_MSE_12_inf_case():
#     """
#     Test the MSE function with infinite values.
#     The MSE should raise a ValueError when the input arrays contain infinite values.
#     """
#     X = np.array([1.0, 2.0, np.inf])
#     y_pred = np.array([1.0, 2.0, 3.0])
#     with pytest.raises(ValueError):
#         MSE(X, y_pred)

