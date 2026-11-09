import numpy as np

class ClassificationMetrics():
    def __init__(self, y_true : np.ndarray, y_pred : np.ndarray, sample_weights : np.ndarray = None, labels : list = None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weights = sample_weights
        self.labels = labels

    def score(self, measure, average = "binary"):
        if measure == "f1":
            pass
        if measure == "precision":
            pass
        if measure == "recall":
            pass

    def accuracy(self, matrix = True, normalize = None):
        y_true = self.y_true
        y_pred = self.y_pred
        n = len(y_true)
        l = len(self.labels)
        if matrix == False:
            s = 0
            for i in range(n):
                if y_true[i] == y_pred[i]:
                    s = s + 1
            if normalize == False:
                return s
            else:
                return s / n
        else:
            matrix = np.zeros((l,l), dtype = int)
            np.add.at(matrix, (y_true, y_pred), 1)
            if normalize == "lines":
                s = matrix.sum(axis = 1, keepdims = True)
            elif normalize == "columns":
                s = matrix.sum(axis = 0, keepdims = True)
            elif normalize == False:
                return matrix
            else:
                s = n
            matrix = np.divide(matrix, s, out = np.zeros_like(matrix, dtype=float), where=np.asarray(s) != 0)
            return matrix

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
    loss = ClassificationMetrics([0, 0, 1, 1], [0.25, 0.75, 0.25, 0.75])
    print(loss.loss())