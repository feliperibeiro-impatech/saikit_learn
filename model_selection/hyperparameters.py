import numpy as np

class GridSearchCV():
    def __init__(self, estimator, param_grid : dict, scoring = None, refit = True, cv = 5):
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        n_params = []
        for v in values:
            n_params.append(len(v))
        for indices in np.ndindex(*n_params):
            params = {}
            for i in range(len(keys)):
                params[keys[i]] = values[i][indices[i]]
        self.estimator = estimator
        self.param_grid = params
        self.scoring = scoring
        self.refit = refit
        self.cv = 5

    def fit(self, x, y):
        raise NotImplementedError
        self.estimator.fit()
        self.estimator.predict()