import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.uniform(0, 10, 100)
true_m = 2
true_b = 1
noise = np.random.normal(0, 2, 100)
y = true_m * x + true_b + noise

m = 0
b = 0

epochs = 1000
learning_rate = 0.0001

for epoch in range(epochs):
    y_pred = b + x * m
    gradient_m = (2/x.shape[0]) * np.sum((y_pred - y) * x)
    gradient_b = (2/x.shape[0]) * np.sum((y_pred - y) * 1)
    m -= learning_rate * gradient_m
    b -= learning_rate * gradient_b

x_plot = np.linspace(np.min(x), np.max(x), 100)
y_pred = b + x_plot * m

plt.scatter(x, y)
plt.plot(x_plot, y_pred, color='red')
plt.savefig('hw2/linear_regression.png')