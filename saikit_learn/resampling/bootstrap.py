import numpy as np

def bootstrap(data : np.ndarray, n_resamples : int, sample_size = 0) -> np.ndarray:
    if sample_size < 1:
        sample_size = np.size(data)
    return np.random.choice(data, (n_resamples, sample_size))

if __name__ == "__main__":
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(bootstrap(data, 10, 5))
    print(bootstrap(data, 10))