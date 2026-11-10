import numpy as np
from saikit_learn.resampling.cross_validation import cross_validation

class GridSearchCV():
    def __init__(self, estimator, param_grid : dict, scoring, refit = True, cv = 5):
        """Inicializa o grid search

        Args:
            estimator (func): Modelo a ter hiperparametros testados.
            param_grid (dict): Grade de parametros a serem testados.
            scoring (_type_): Métrica para decisão do melhor modelo.
            refit (bool, optional): Define saída da função fit. Padroniza para True.
            cv (int, optional): Número de folds. Padroniza para 5.
        """
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

    def fit(self, x : np.ndarray, y : np.ndarray):
        """Treina os modelos e decide qual é o melhor.

        Args:
            x (np.ndarray): Atributos de entrada
            y (np.ndarray): Valores alvo

        Returns:
            dict: Planilha de resultados
            float: Pontuação do melhor modelo
            int: índice do melhor modelo
            dict: Parametros do melhor modelo
            func: Modelo treinado ou não a depender de refit.
        """
        estimator = self.estimator
        param_grid = self.param_grid
        k = self.cv
        grid = []
        results = {}
        for key in param_grid[0].keys():
            results[key] = []
            for params in param_grid:
                results[key].append(params[key])
        for params in param_grid:
            grid.append(estimator(**params))
        results["score"] = []
        for instance in grid:
            scores = cross_validation(instance, x, y, self.scoring, k)
            results["score"].append(np.mean(scores))
        aux = np.argsort(results["score"])[::-1]
        best_index = aux[0]
        best_score = results["score"][best_index]
        best_params = param_grid[best_index]
        best_estimator = estimator(**best_params)
        ranking = np.empty_like(aux)
        ranking[aux] = np.arange(1, len(aux) + 1)
        results["ranking"] = ranking
        if self.refit == False:
            return results, best_score, best_index, best_params, best_estimator
        else:
            return best_estimator.fit(x, y)