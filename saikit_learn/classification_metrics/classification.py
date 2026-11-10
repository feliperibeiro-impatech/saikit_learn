import numpy as np

class ClassificationMetrics():
    def __init__(self, y_true : np.ndarray, y_pred : np.ndarray, labels : list = None, sample_weights : np.ndarray = None):
        """Iniciliza a classe classification metrics

        Args:
            y_true (np.ndarray): Classes corretas.
            y_pred (np.ndarray): Classes previtas.
            labels (list, opcional): Classes. Se não for fornecido induz de y_true e y_pred
            sample_weights (np.ndarray, opcional): Pesos das amostras. Se não for fornecido padroniza para uma array de uns.

        Raises:
            ValueError: labels tem menos classes que y_true
            ValueError: y_true ou y_pred tem mais de uma dimensão
            ValueError: y_true e y_pred tem formatos diferentes
            ValueError: sample_weights tem formato diferente de y_true
            ValueError: sample_weigths tem pesos negativos
        """
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_pred)))
        unknown = np.setdiff1d(np.concatenate((y_true, y_pred)), labels)
        if unknown.size > 0:
            raise ValueError(f"y_true/y_pred contêm valores ausentes em labels: {unknown.tolist()}.")
        label_to_index = {label: i for i, label in enumerate(labels)}
        self.labels = labels
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("y_true e y_pred devem ser vetores 1D.")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true e y_pred devem ter o mesmo formato.")
        self.y_true = np.array([label_to_index[v] for v in y_true])
        self.y_pred = np.array([label_to_index[v] for v in y_pred])
        if sample_weights is None:
            sample_weights = np.ones(len(y_true))
        if sample_weights.shape != y_true.shape:
            raise ValueError("sample_weights deve ter o mesmo formato de y_true.")
        if np.any(sample_weights < 0):
            raise ValueError("sample_weights não pode conter valores negativos.")
        self.sample_weights = sample_weights

    def confusion_matrix(self, normalize = "None") -> np.ndarray:
        """Gera a matriz de confusão

        Args:
            normalize (str, opcional): Define a normalização da matriz. Possíveis valores: "lines", "columns", "all". Se não for fornecido Não normaliza.

        Returns:
            np.ndarray: Matriz de confusão
        """
        y_true = self.y_true
        y_pred = self.y_pred
        sample_weights = self.sample_weights
        l = len(self.labels)
        matrix = np.zeros((l,l), dtype = float)
        np.add.at(matrix, (y_true, y_pred), sample_weights)
        if normalize == "lines":
            s = matrix.sum(axis = 1, keepdims = True)
        elif normalize == "columns":
            s = matrix.sum(axis = 0, keepdims = True)
        elif normalize == "all":
            s = sample_weights.sum()
        else:
            return matrix
        matrix = np.divide(matrix, s, out = np.zeros_like(matrix, dtype=float), where=np.asarray(s) != 0)
        return matrix

    def score(self, measure : str, average = "binary", pos_label = 1, normalize = True) -> float:
        """Pontua uma classificação

        Args:
            measure (str): Medida. Possíveis valores: "accuracy", "precision", "recall", "f1".
            average (str, opcional): Média para classificações multivariadas. Possíveis valores: "micro", "macro". Se não for fornecida padroniza para classificação binária.
            pos_label (int, opcional): índice da classe positiva. Padroniza para 1.
            normalize (bool, opcional): Define a normalização da accuracy. Padroniza para True.

        Raises:
            ValueError: Medida inválida
            ValueError: Média inválida
            ValueError: Binary para multivariádas
            TypeError: pos_labels não é indice de labels

        Returns:
            float: Pontuação
        """
        y_true = self.y_true
        y_pred = self.y_pred
        sample_weights = self.sample_weights
        n = len(y_true)
        w = sample_weights.sum()
        if measure not in ("accuracy", "precision", "recall", "f1"):
            raise ValueError("Medida inválida")
        if measure == "accuracy":
                    s = 0
                    for i in range(n):
                        if y_true[i] == y_pred[i]:
                            s = s + sample_weights[i]
                    if normalize:
                        return s / w
                    else:
                        return s
        matrix = self.confusion_matrix()
        tp = np.diag(matrix)
        fn = matrix.sum(axis = 1) - tp
        fp = matrix.sum(axis = 0) - tp
        if average not in ("binary", "micro", "macro"):
            raise ValueError("Média inválida")
        if average == "binary":
            if len(self.labels) != 2:
                raise ValueError("average='binary' exige duas classes.")
            if not isinstance(pos_label, (int, np.integer)):
                raise TypeError("pos_label deve ser um indice de labels")
            tp = tp[pos_label]
            fn = fn[pos_label]
            fp = fp[pos_label]
        if average == "micro":
            tp = tp.sum()
            fn = fn.sum()
            fp = fp.sum()
        if average == "macro":
            if measure == "precision":
                score = np.divide(tp, tp + fp, out = np.zeros_like(tp, dtype = float), where = (tp + fp) != 0)
            elif measure == "recall":
                score = np.divide(tp, tp + fn, out = np.zeros_like(tp, dtype = float),where = (tp + fn) != 0)
            elif measure == "f1":
                score = np.divide(2 * tp, 2 * tp + fp + fn, out = np.zeros_like(tp, dtype = float), where = (2 * tp + fp + fn) != 0)
            return np.mean(score)
        if tp == 0:
            return 0
        if measure == "precision":
            return tp / (tp + fp)
        elif measure == "recall":
            return tp / (tp + fn)
        elif measure == "f1":
            return 2 * tp / (2 * tp + fp + fn)

    def loss(self, normalize = True) -> float:
        """Gera a pontuação log_loss

        Args:
            normalize (bool, opcional): Define a normalização. Padroniza para True.

        Raises:
            ValueError: y_prob não é uma probabilidade

        Returns:
            float: Pontuação
        """
        y_true = self.y_true
        y_prob = self.y_pred
        if np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError("y_prob não é uma probabilidade")
        sample_weights = self.sample_weights
        loss = -np.sum(sample_weights * (y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
        if normalize:
            return loss / sample_weights.sum()
        else:
            return loss