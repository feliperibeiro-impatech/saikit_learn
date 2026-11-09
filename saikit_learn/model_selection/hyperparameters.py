import numpy as np

class GridSearchCV():
    def __init__(self, estimator, param_grid : dict, scoring = None, refit = True, cv = 5, ranking = "test"):
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        n_params = []
        param_grid = []
        for v in values:
            n_params.append(len(v))
        for indices in np.ndindex(*n_params):
            params = {}
            for i in range(len(keys)):
                params[keys[i]] = values[i][indices[i]]
            param_grid.append(params)
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.refit = refit
        self.cv = cv
        self.ranking = ranking

    def fit(self, x, y):
        estimator = self.estimator
        param_grid = self.param_grid
        score = self.scoring
        grid = []
        results = {}
        for key in self.param_grid[0].keys():
            results[key] = []
            for params in self.param_grid:
                results[key].append(params[key])
        results["score_train"] = []
        results["score_test"] = []
        #cross validation
        for instance in grid:
            instance.fit(x_train, y_train)
            pred_train = instance.predict(x_train)
            pred_test = instance.predict(x_test)
            results["score_train"].append(score(y_train, pred_train))
            results["score_test"].append(score(y_test, pred_test))
        aux = np.argsort(results["score_test"])[::-1]
        best_index = aux[0]
        best_score = results["score_test"][best_index]
        ranking = np.empty_like(aux)
        ranking[aux] = np.arange(1, len(aux) + 1)
        results["ranking"] = ranking
        if self.refit == False:
            return results, best_score, best_index