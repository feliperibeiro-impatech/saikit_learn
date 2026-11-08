from saikit_learn.regressao.minimos_quadrados import MQO
import numpy as np

n, p = 100, 1
x = np.random.random((n, p))

def f(t):
    return - np.pi * t.ravel() + 1/2

eps = np.random.normal(scale = 0.5, size = n)
y = f(x) + eps

# treino
model = MQO()
model.train(x, y)
print(model.beta)

x_n = np.linspace(0, 1, 200).reshape(-1, 1)
f_hat = model.predict(x_n)

# definir testes...

def teste1():
    assert 1 == 1


