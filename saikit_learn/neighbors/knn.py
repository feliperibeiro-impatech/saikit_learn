import numpy as np

class KNN:
    def __init__(self, k: int, p: int = 2) -> None:
        """Inicializa o classificador KNN.

        Args:
            k (int): Número de base para a comparação
            p (int, opcional): Parâmetro que define a métrica da distância. Default: 2.
        """     

        if k < 1:
            raise ValueError("k deve ser maior ou igual a 1.")
        if p not in [1,2]:
            raise ValueError("p deve ser 1 ou 2.")
        
        self._k = k
        self._p = p
        self._X = None
        self._y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina o modelo de classificação KNN.

        Args:
            X (np.ndarray): Matriz dos atributos
            y (np.ndarray): Vetor com os resultados

        Returns:
            self: KNN
        """   

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X deve ser np.ndarray, recebeu {type(X).__name__}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y deve ser np.ndarray, recebeu {type(y).__name__}")
        
        if len(X) != len(y):
            raise ValueError("X e y devem ser de mesmo tamanho") 

        self._X = X
        self._y = y

        return self

    def _get_distances_with_labels(self, x: np.ndarray) -> np.ndarray:
        """Ordena os rótulos de treino pela distância até um ponto.

        Args:
            x (np.ndarray): Ponto a ser comparado.

        Returns:
            np.ndarray:  Rótulos de treino ordenados por distância crescente
            até x.
        """    

        diff = np.abs(self._X - x)
        distances = np.sum(diff ** self._p, axis=1) ** (1 / self._p)

        result = np.argsort(distances, kind='stable')
        
        return self._y[result]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Estima a classificação de cada rótulo pelos k vizinhos mais próximos.

        Args:
            X (np.ndarray): Amostras pendentes de classificação.

        Returns:
            np.ndarray: Resultado da estimativa. 
        """        

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X deve ser np.ndarray, recebeu {type(X).__name__}")        
        
        predicted_labels = []

        for x in X:
            sorted_labels = self._get_distances_with_labels(x)

            k_neighbors = sorted_labels[:self._k]

            # Escolhe o primero rótulo por ser a dos pontos mais próximos
            label, first_pos, count = np.unique(k_neighbors, return_index=True, return_counts=True)
            candidates = count == count.max()
            predicted_labels.append(label[np.argmin(first_pos[candidates])])

        return np.array(predicted_labels)
