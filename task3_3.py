import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Download data from data-logistic.csv file.
df= pd.read_csv('data-logistic.csv', header=None)
y = df[0].values
X = df[[1, 2]].values

# Implement gradient descent for regular and L2-regularized (C=10) logistic regression. k=0.1, w=(0,0).
def update_w1(w1, w2, X, y, k, C):

    s = 0.
    l = len(y)
    for y, X in zip(y, X):
        s += y*X[0]*(1.0 - 1.0/(1.0 + np.exp(-y*(w1*X[0] + w2*X[1]))))
    return w1 + k*(s/l - C*w1)


def update_w2(w1, w2, X, y, k, C):

    s = 0.
    l = len(y)
    for y, X in zip(y, X):
        s += y*X[1]*(1.0 - 1.0/(1.0 + np.exp(-y*(w1*X[0] + w2*X[1]))))
    return w2 + k*(s/l - C*w2)


def sigm(X, w1, w2):
    return 1 / (1 + np.exp(-w1 * X[:, 0] - w2 * X[:, 1]))


def GD(X, y, w1=0, w2=0, k=0.1, C=0, err=1e-5, max_iter=10000):

    w1_new, w2_new = w1, w2
    i = 0
    for i in range(max_iter):
        w1_new = update_w1(w1, w2, X, y, k, C)
        w2_new = update_w2(w1, w2, X, y, k, C)
        error = np.sqrt((w1_new - w1)**2 + (w2_new - w2)**2)
        if error <= err:
            break
        else:
            w1, w2 = w1_new, w2_new

    return w1_new, w2_new, i+1

# Run gradient descent and bring to convergence.
# What is the AUC-ROC value in learning without regularization and when using it?
w1, w2, n_iter = GD(X, y)
a = sigm(X, w1, w2)
print(f"{roc_auc_score(y, a):0.3f}", '\nn_ite r =', n_iter)

w1, w2, n_iter = GD(X, y, C=10)
a = sigm(X, w1, w2)
print(f"{roc_auc_score(y, a):0.3f}", '\nn_iter =', n_iter)















