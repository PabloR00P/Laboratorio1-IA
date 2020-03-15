import numpy as np
import pandas
from matplotlib import pyplot as plt
from linear_cost import linear_cost
from linear_cost_derivate import linear_cost_derivate
from gradient_descent import gradient_descent

df = pandas.read_csv('Admission_Predict.csv', usecols=['GRE Score', 'TOEFL Score', 'CGPA', 'Chance of Admit'])
dataset = np.array(df)

np.random.shuffle(dataset)
training_set, validation_set, test_set = dataset[:240,:], dataset[240:320,:], dataset[320:,:]

x1 = training_set.T[0]
x2 = training_set.T[1]
x3 = training_set.T[2]
x3cv = validation_set.T[2]
y = training_set.T[3]
ycv = validation_set.T[3]

X = np.vstack(
    (
    	np.ones(240),
        x3
    )
).T

Xcv = np.vstack(
    (
    	np.ones(80),
        x3cv
    )
).T

theta_0 = np.random.rand(2, 1)
y = y.reshape(240, 1)
ycv = ycv.reshape(80, 1)

theta, costs, gradient_norms = gradient_descent(
    X,
    y,
    theta_0,
    linear_cost,
    linear_cost_derivate,
    alpha=0.0001,
    treshold=0.0001,
    max_iter=100000,
    l=8
)

plotArray = np.arange(min(x3),11)
l = plotArray.shape

def r2_score_from_scratch(ys_orig, ys_line):
    y_mean_line = [ys_orig.mean() for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

#r_squared = r2_score_from_scratch(y, yr)


cost_train = []
cost_cv = []

for i in range(0, 2, 1):
    cost_train.append(linear_cost(X ,y,theta, 7))
    cost_cv.append(linear_cost(Xcv ,ycv,theta, 7))

#plt.plot(np.arange(0, 2, 1), cost_train);
#plt.plot(np.arange(0, 2, 1), cost_cv, color="green");


#plt.scatter(x3, y)
#plt.scatter(x3, np.matmul(X,theta))
plt.plot(np.arange(len(costs)), costs)
#plt.plot(plotArray, np.matmul(np.vstack((np.ones(l),plotArray,plotArray**2)).T, theta), color='red')
plt.show()