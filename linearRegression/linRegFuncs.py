import numpy as np
from scipy import optimize

def costFunction(theta, X, y, lambda_):
    m = len(y)  # number of training examples
    J = 0
    grad = np.zeros_like(theta)
    h = X @ theta.T

    reg = theta  #regularisation initilised to theta (to then minimise)
    reg[0] = 0       #preventing bias term from being regularised

    J = (1 / m) * np.sum((h - y)**2)
    J = J + (lambda_/(2*m)) * np.sum( np.square(reg) )
    grad = (1 / m) * ((h - y) @ X)
    grad = grad + (lambda_/m) * reg
    print(grad)
    return J, grad

def trainModel(X, y, lambda_ = 0, maxiter = 400):
    theta_init = np.zeros(X.shape[1])

    options = {"maxiter": maxiter}

    res = optimize.minimize(costFunction, theta_init, (X, y, lambda_), jac = True, method = "TNC", options = options)

    return res.x

def predict(theta,X):
    m,n = np.shape(X)
    p = np.zeros(m)
    p = X @ theta.T
    return p

def learningCurve(X, y, X_CV, y_CV, lambda_):
    m = y.size
    error_train = np.zeros(m)
    error_CV = np.zeros(m)

    for i in range(m):
        X_train = X[0:i+1,:]
        y_train = y[0:i+1]
        
        theta = trainModel(X_train, y_train)

        error_train[i], _ = costFunction(theta, X_train, y_train, lambda_)
        error_CV[i], _ = costFunction(theta, X_CV, y_CV, lambda_)

    return error_train, error_CV

def validationCurve(X, y, X_CV, y_CV, lambdas):
    n = len(lambdas)
    m = y.size

    error_train = np.zeros(n)
    error_CV = np.zeros(n)

    for i in range(n):
        lambda_try = lambdas[i]
        theta = trainModel(X, y, lambda_ = lambda_try)
        error_train[i], _ = costFunction(theta, X, y, lambda_ = 0)
        error_CV[i], _ = costFunction(theta, X_CV, y_CV, lambda_ = 0)
    return error_train, error_CV