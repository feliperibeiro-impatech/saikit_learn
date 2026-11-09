import numpy as np
from saikit_learn.neural_network import MLPRegressor

def perda(modelo, X, Y):
    activations, _ = modelo._forward(X)
    return np.mean((activations[-1] - Y.reshape(-1, 1)) ** 2) / 2

rng = np.random.default_rng(0)
X, Y = rng.normal(size=(7, 3)), rng.normal(size=7)

modelo = MLPRegressor(hidden_layers=(4, 3), random_state=0)
modelo._init_parameters([3, 4, 3, 1])

activations, zs = modelo._forward(X)
gW, gb = modelo._backward(Y.reshape(-1, 1), activations, zs)

eps = 1e-5
for nome, params, grads in (("W", modelo._weights, gW), ("b", modelo._biases, gb)):
    for l, (p, g) in enumerate(zip(params, grads)):
        erro = 0.0
        for idx in np.ndindex(p.shape):
            o = p[idx]
            p[idx] = o + eps; mais = perda(modelo, X, Y)
            p[idx] = o - eps; menos = perda(modelo, X, Y)
            p[idx] = o
            erro = max(erro, abs(g[idx] - (mais - menos) / (2 * eps)))
        print(nome, "camada", l, "erro máximo", erro)

print(np.abs(zs[1]).min())