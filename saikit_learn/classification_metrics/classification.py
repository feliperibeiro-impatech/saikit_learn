import numpy as np

class ClassificationMetrics():
    def __init__(self, y_true : np.ndarray, y_pred : np.ndarray, sample_weights : np.ndarray = None, labels : list = None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weights = sample_weights
        self.labels = labels

    def confusion_matrix(self, normalize = "None"):
        y_true = self.y_true
        y_pred = self.y_pred
        l = len(self.labels)
        matrix = np.zeros((l,l), dtype = int)
        np.add.at(matrix, (y_true, y_pred), 1)
        if normalize == "lines":
            s = matrix.sum(axis = 1, keepdims = True)
        elif normalize == "columns":
            s = matrix.sum(axis = 0, keepdims = True)
        elif normalize == "all":
            s = len(y_true)
        else:
            return matrix
        matrix = np.divide(matrix, s, out = np.zeros_like(matrix, dtype=float), where=np.asarray(s) != 0)
        return matrix

    def score(self, measure, average = "binary", normalize = True):
        y_true = self.y_true
        y_pred = self.y_pred
        n = len(y_true)
        if measure == "accuracy":
                    s = 0
                    for i in range(n):
                        if y_true[i] == y_pred[i]:
                            s = s + 1
                    if normalize:
                        return s / n
                    else:
                        return s
        matrix = self.confusion_matrix()
        tp = matrix[1][1]
        fn = matrix[1][0]
        fp = matrix[0][1]
        if measure == "precision":
            return tp / (tp + fp)
        if measure == "recall":
            return tp / (tp + fn)
        if measure == "f1":
            return 2 * tp / (2 * tp + fp + fn)

    def loss(self, normalize = True, custom = False) -> float:
        if custom == True:
            raise NotImplementedError
        else:
            y_true = self.y_true
            y_prob = self.y_pred
            n = len(y_true)
            one = np.ones(n)
            loss = -(np.dot(y_true, np.log(y_prob)) + np.dot(one - y_true, np.log(one - y_prob)))
            if normalize == False:
                return loss
            else:
                return loss / n

if __name__ == "__main__":
    y_true = [0, 0, 1, 1]
    loss = ClassificationMetrics(y_true, [0.25, 0.75, 0.25, 0.75])
    print(loss.loss())
    score = ClassificationMetrics(y_true, [0, 0, 0, 0], labels = [0, 1])
    matrix = score.confusion_matrix()
    print(matrix)