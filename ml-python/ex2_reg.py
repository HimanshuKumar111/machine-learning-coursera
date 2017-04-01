#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import newaxis, r_, c_, mat, e
from numpy.linalg import *

def plotData(X, y):
    pos = (y.ravel() == 1).nonzero()
    neg = (y.ravel() == 0).nonzero()

    plt.plot(X[pos, 0].T, X[pos, 1].T, 'k+', linewidth=2, markersize=9, markeredgewidth=2)
    plt.plot(X[neg, 0].T, X[neg, 1].T, 'ko', markerfacecolor='r', markersize=7)

def mapFeature(X1, X2):
    X1 = mat(X1); X2 = mat(X2)

    degree = 6
    out = [np.ones(X1.shape[0])]
    for i in xrange(1, degree+1):
        for j in xrange(0, i+1):
            #out = c_[out, X1.A**(i-j) * X2.A**j] # too slow, what's numpy way?
            out.append(X1.A**(i-j) * X2.A**j)
    return mat(out).T

def sigmoid(z):
    g = 1. / (1 + e**(-z.A))
    return g

def costFunctionReg(theta, X, y, lmd):
    m = X.shape[0]
    predictions = sigmoid(X * c_[theta])

    J = 1./m * (-y.T.dot(np.log(predictions)) - (1-y).T.dot(np.log(1 - predictions)))
    J_reg = lmd/(2*m) * (theta[1:] ** 2).sum()
    J += J_reg

    grad0 = 1/m * X.T[0] * (predictions - y)
    grad = 1/m * (X.T[1:] * (predictions - y) + lmd * c_[theta[1:]])
    grad = r_[grad0, grad]
    return J[0][0]##, grad

def predict(theta, X):
    p = sigmoid(X * c_[theta]) >= 0.5
    return p

def plotDecisionBoundary(theta, X, y):
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        plot_x = r_[X[:,2].min()-2,  X[:,2].max()+2]
        plot_y = (-1./theta[2]) * (theta[1]*plot_x + theta[0])

        plt.plot(plot_x, plot_y)
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        for i in xrange(0, len(u)):
            for j in xrange(0, len(v)):
                z[i, j] = mapFeature(u[i], v[j]) * c_[theta]
        z = z.T

        plt.contour(u, v, z, [0, 0], linewidth=1)


if __name__ == '__main__':
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = mat(c_[data[:, :2]])
    y = c_[data[:, 2]]

    plotData(X, y)

    plt.ylabel('Microchip test 1')
    plt.xlabel('Microchip test 2')
    plt.legend(['y = 0', 'y = 1'])
    plt.show()

    # ========== Part 1: Regularized Logistic Regression

    X = mapFeature(X[:, 0], X[:, 1])

    initial_theta = np.zeros(X.shape[1])

    lmd = 1

    ## cost, grad = ...
    cost = costFunctionReg(initial_theta, X, y, lmd)

    print 'Cost at initial theta (zeros):', cost

    raw_input('Press any key to continue\n')

    # ========== Part 2: Regularization and Accuracies

    initial_theta = np.zeros(X.shape[1])

    lmd = 1

    #options = {'full_output': True, 'maxiter': 400} # fmin
    options = {'full_output': True} # fmin_powell

    theta, cost, _, _, _, _ = \
        optimize.fmin_powell(lambda t: costFunctionReg(t, X, y, lmd),
                                initial_theta, **options)

    plotDecisionBoundary(theta, X, y)

    plt.title('lambda = %s' % lmd)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    plt.show()

    p = predict(theta, X);

    print 'Train Accuracy:', (p == y).mean() * 100

    raw_input('Press any key to continue\n')
