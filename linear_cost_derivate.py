import numpy as np

def linear_cost_derivate(X, y, theta, l):
    h = np.matmul(X, theta)
    m, _ = X.shape
    return np.matmul((h - y).T, X).T / m + ((l / m) * theta.sum())