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
    grad = (1 / m) * (X.T  @ (h - y))
    grad = grad + (lambda_/m) * reg

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

def learningCurve(X, y, X_CV, y_CV, lambda_, step=1):
    m = y.size
    error_train = []
    error_CV = []

    for i in range(0, m, step):
        X_train = X[0:i+1,:]
        y_train = y[0:i+1]
        
        theta = trainModel(X_train, y_train)

        error_train_temp, _ = costFunction(theta, X_train, y_train, lambda_ = 0)
        error_CV_temp, _ = costFunction(theta, X_CV, y_CV, lambda_ = 0)

        p_train = predict(theta, X)
        p_CV = predict(theta, X_CV)

        #error_train_temp = np.mean( np.abs(y-p_train)/y)
        #error_CV_temp = np.mean( np.abs(y_CV-p_CV)/y_CV)

        error_train.append(error_train_temp)
        error_CV.append(error_CV_temp)
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
        p_train = predict(theta, X)
        p_CV = predict(theta, X_CV)
        
        #error_train[i] = np.mean( np.abs(y-p_train)/y)
        #error_CV[i] = np.mean( np.abs(y_CV-p_CV)/y_CV)

    return error_train, error_CV

def featureNormalisation(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma

def reverseNormalisation(X, mu, sigma):
    X = X * sigma + mu
    return X