import numpy as np

def verify_array(X, name='X'):
    '''Converts the entry to a ndarray float 2D and verify its content
    
    Raises:

        ValueError: If X is not 2D, or is empty or contains NaN/inf.
    '''
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            f'{name} must be 2D (n_sample, n_features); receive {X.ndim}D. '
            'If there is only one feature, use X.reshape(-1, 1).'
            )

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(
            f'{name} is empty; shape {X.shape}.'
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            f'{name} contains NaN or inf.'
        )
    return X

def verify_X_y(X, y):
    '''Verify the content of the ndarrays and validate if they have the same sample size'''
    X = verify_array(X, "X")
    y = np.asarray(y, dtype=float)
    # if y is a column matrix, converts to an column vector
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()

    if y.ndim != 1:
        raise ValueError(
            f'y must be 1D; receive {y.ndim}D.'
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            f'y contains NaN or inf'
            )
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f'X and y must have the same sample number; {X.shape[0]} != {y.shape[0]}'
        )

    return X, y

class NotFittedError(ValueError, AttributeError):
    '''Raised when use a model (predict) before call fit'''
