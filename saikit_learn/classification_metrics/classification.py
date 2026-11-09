import numpy as np

class ClassificationMetrics():
    def __init__(self, y_true : np.ndarray, y_pred : np.ndarray, labels : list = None, sample_weights : np.ndarray = None):
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_pred)))
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

    def confusion_matrix(self, normalize = "None"):
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

    def score(self, measure, average = "binary", pos_label = 1, normalize = True) -> float:
        y_true = self.y_true
        y_pred = self.y_pred
        sample_weights = self.sample_weights
        n = len(y_true)
        w = sample_weights.sum()
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
        if average == "binary":
            if len(self.labels) != 2:
                raise ValueError("average='binary' exige duas classes.")
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
        if measure == "precision":
            return tp / (tp + fp)
        if measure == "recall":
            return tp / (tp + fn)
        if measure == "f1":
            return 2 * tp / (2 * tp + fp + fn)

    def loss(self, normalize = True) -> float:
        y_true = self.y_true
        y_prob = self.y_pred
        sample_weights = self.sample_weights
        loss = -np.sum(sample_weights * (y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
        if normalize:
            return loss / sample_weights.sum()
        else:
            return loss

if __name__ == "__main__":
    y_true = [0, 0, 1, 1]
    loss = ClassificationMetrics(y_true, [0.25, 0.75, 0.25, 0.75])
    print(loss.loss())
    score = ClassificationMetrics(y_true, [0, 0, 0, 0], labels = [0, 1])
    matrix = score.confusion_matrix()
    print(matrix)