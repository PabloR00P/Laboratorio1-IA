import numpy as np

def gradient_descent(
        X,
        y,
        theta_0,
        cost,
        cost_derivate,
        alpha=0.01,
        l=0,
        treshold=0.0001,
        max_iter=10000):
    theta, i = theta_0, 0
    costs = []
    gradient_norms = []
    while np.linalg.norm(cost_derivate(X, y, theta, l)) > treshold and i < max_iter:
        theta -= alpha * cost_derivate(X, y, theta, l)
        i += 1
        costs.append(cost(X, y, theta, l))
        gradient_norms.append(cost_derivate(X, y, theta, l))
    return theta, costs, gradient_norms