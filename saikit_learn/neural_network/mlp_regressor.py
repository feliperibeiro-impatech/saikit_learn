import numpy as np

class MLPRegressor:
    def __init__(self, hidden_layers: tuple=(10,), learning_rate: float=0.01, n_epochs: int=1000, random_state: int = None):
        """Inicializa o regressor MLP.

        Args:
            hidden_layers (tuple, opcional): Tamanhos das camadas ocultas. Padrão é (10,).
            learning_rate (float, opcional): Passo do gradiente descende. Padrão é 0.01.
            n_epochs (int, opcional): Vezes que o conjunto de treino é percorrido. Padrão é 1000.
            random_state (int, opcional): Seed geradora dos pesos inciais. Padrão é None.
        """
        
        if not isinstance(hidden_layers, (tuple, list)):
            raise TypeError("hidden_layers deve ser uma tupla de inteiros.")
        
        if any((not isinstance(h, (int, np.integer))) or h <= 0 for h in hidden_layers):
            raise ValueError("Cada camada oculta deve ter um número inteiro positivo de neurônios.")
        
        if learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo.")
        
        if not isinstance(n_epochs, (int, np.integer)) or n_epochs <= 0:
            raise TypeError("n_epochs deve ser um inteiro positivo.")

        self._hidden = hidden_layers
        self._lr = learning_rate
        self._n_epochs = n_epochs
        self._rng = np.random.default_rng(random_state)
        self._weights, self._biases, self._n_features = None, None, None

    def _init_parameters(self, sizes: list):
        """Inicializa os pesos e vieses de cada camada.

        Args:
            sizes (list): Matriz com o tamanho das camadas de entrada, oculta e de saída.
        """        
        self._weights, self._biases = [], [] 
        for layer in range(len(sizes)-1):
            n_in, n_out = sizes[layer], sizes[layer + 1]

            # Desvio He para ReLU
            W = self._rng.normal(0, np.sqrt(2/n_in), size=(n_in, n_out))   
            b = np.zeros(n_out)
            self._weights.append(W)
            self._biases.append(b)

    def _forward(self, X: np.ndarray) ->  tuple[np.ndarray, np.ndarray]:
        """Propaga o X pela rede neural. 

        Args:
            X (np.ndarray): _description_

        Returns:
            np.ndarray: Lista com L+1 arrays; activations[0] é X e activations[-1] é a saída (n, 1).
            np.ndarray: Lista com L arrays com os valores antes da ativação de cada camada.
        """        
        activations, zs = [X], []
        n_layers = len(self._weights)

        for l in range(n_layers):
            z = activations[-1] @ self._weights[l] + self._biases[l]
            zs.append(z)

            if l < n_layers- 1:
                a = np.maximum(0, z)
            else:
                a = z

            activations.append(a)

        return activations, zs

    def _backward(self, y: np.ndarray, activations: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calcula os gradientes da perda em relação a pesos e vieses.

        Args:
            y (np.ndarray): _description_
            activations (np.ndarray): _description_
            z (np.ndarray): _description_

        Returns:
            np.ndarray:Gradientes dos pesos, com os mesmos tamanhos de self._weights.
            np.ndarray: Gradientes dos vieses, com os mesmos tamanhos de self._biases.
        """
        n_samples = y.shape[0]    
        n_layers = len(self._weights)

        y_pred = activations[-1]
        delta = (y_pred-y) / n_samples
        grads_W, grads_b = [None] * n_layers, [None] * n_layers

        for l in reversed(range(n_layers)):
            grads_W[l] = activations[l].T @ delta
            grads_b[l] = np.sum(delta, axis=0)
            if l > 0: 
                reLU_deriv = (z[l - 1] > 0)
                delta = delta @ self._weights[l].T * reLU_deriv

        return grads_W, grads_b

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina o regressor MLP por gradiente descendente.

        Args:
            X (np.ndarray): Atributos de treino.
            y (np.ndarray): Valores-alvo contínuos.
        """      
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X deve ser np.ndarray, recebeu {type(X).__name__}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y deve ser np.ndarray, recebeu {type(y).__name__}")
            
        if X.ndim != 2:
            raise ValueError("X deve ser 2D")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y deve ser 1D e ter o mesmo número de amostras que X.")
        if X.shape[0] == 0:
            raise ValueError("X não pode ser vazio.")
        if not np.isfinite(X).all():
            raise ValueError("X contém NaN ou infinito.")
        if not np.isfinite(y).all():
            raise ValueError("y contém NaN ou infinito.")

        n_samples, self._n_features = X.shape
        y = y.reshape(-1, 1)

        size = [self._n_features, *self._hidden, 1]
        self._init_parameters(size)

        self._loss_history = []
        for epoch in range(self._n_epochs):
            activations, zs = self._forward(X)
            y_pred = activations[-1]

            loss = np.mean((y_pred - y)**2) / 2

            if not np.isfinite(loss):
                raise FloatingPointError("A perda divergiu; reduza learning_rate ou padronize X.")
            self._loss_history.append(loss)

            grads_W, grads_b = self._backward(y, activations, zs)

            for l in range(len(self._weights)):
                self._weights[l] -= self._lr * grads_W[l]
                self._biases[l] -= self._lr * grads_b[l]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz o valor contínuo de cada amostra.

        Args:
            X (np.ndarray): Valores a prever, de tamanho (n_samples, n_features).

        Returns:
            np.ndarray: Valores preditos, de tamanho (n_samples,).
        """        
        if self._weights is None:
            raise RuntimeError("Chame fit antes de predict.")

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X deve ser np.ndarray, recebeu {type(X).__name__}.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError(f"X deve ter shape (n_samples, {self._n_features}).")
        if not np.isfinite(X).all():
            raise ValueError("X contém NaN ou infinito.")

        activations, _ = self._forward(X)
        return activations[-1].ravel()
    
    
    