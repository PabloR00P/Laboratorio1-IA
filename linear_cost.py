import numpy as np

def linear_cost(X, y, theta, l):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    st = theta**2
    return sq.sum() / (2*m) + (l * (st.sum() / (2 * m)))