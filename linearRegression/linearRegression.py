from math import cos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from linRegFuncs import *

"""Number of folds of validation, k, and repeats, r"""
r = 50

"""Sorting all data, including normalisation values"""
all_data = pd.read_csv("./data/train.csv")
all_y = all_data["SalePrice"].to_numpy()
all_X = all_data.drop(["SalePrice"], axis = 1).to_numpy()
all_X, X_mu, X_sigma = featureNormalisation(all_X)
all_y, y_mu, y_sigma = featureNormalisation(all_y)

"""Sorting Validation curve"""
lambdas = np.array([0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
lambda_errors = pd.DataFrame()
lambda_errors["Lambda"] = lambdas

lambda_errors_avg = pd.DataFrame()
lambda_errors_avg["Lambda"] = lambdas
train_mean = np.zeros_like(lambdas)
train_std = np.zeros_like(lambdas)
CV_mean = np.zeros_like(lambdas)
CV_std = np.zeros_like(lambdas)


for i in range(0,r):
    """Sorting CV data"""
    CV =  all_data.sample(frac=0.2)
    y_CV = (CV["SalePrice"].to_numpy()-y_mu)/y_sigma
    X_CV = (CV.drop(["SalePrice"], axis = 1).to_numpy()-X_mu)/X_sigma
    X_CV = np.c_[np.ones(X_CV.shape[0]), X_CV]
    data = all_data.drop(CV.index)
    
    """Sorting training data"""
    train = data
    y = (train["SalePrice"].to_numpy()-y_mu)/y_sigma
    X = (train.drop(["SalePrice"], axis = 1).to_numpy()-X_mu)/X_sigma
    X = np.c_[np.ones(X.shape[0]), X]

    error_train, error_CV = validationCurve(X, y, X_CV, y_CV, lambdas)

    lambda_errors["Train Error"+str(i)] = error_train
    lambda_errors["CV Error"+str(i)] = error_CV

for i in range(0,len(lambdas)):
    train_errors = lambda_errors.loc[i,:][1::2].tolist()
    CV_errors = lambda_errors.loc[i,:][2::2].tolist()
    
    train_mean[i] = np.mean(train_errors)
    CV_mean[i] = np.mean(CV_errors)

    train_std[i] = np.std(train_errors)
    CV_std[i] = np.std(CV_errors)
lambda_errors_avg["train_errors"] = train_mean
lambda_errors_avg["CV_errors"] = CV_mean
print(lambda_errors_avg)

lambda_optimal = 1.5*10**2

plt.semilogx(lambdas, CV_mean, label="CV", c="r")
#plt.semilogx(lambdas, CV_mean+CV_std, label="CV", c="b")
#plt.semilogx(lambdas, CV_mean-CV_std, label="CV", c="b")
plt.semilogx(lambdas, train_mean, label="train")
plt.legend()
plt.show()

"""Sorting Training Curve"""
"""Sorting CV data"""
CV =  all_data.sample(frac=0.2)
y_CV = (CV["SalePrice"].to_numpy()-y_mu)/y_sigma
X_CV = (CV.drop(["SalePrice"], axis = 1).to_numpy()-X_mu)/X_sigma
X_CV = np.c_[np.ones(X_CV.shape[0]), X_CV]
data = all_data.drop(CV.index)
    
"""Sorting training data"""
train = data
y = (train["SalePrice"].to_numpy()-y_mu)/y_sigma
X = (train.drop(["SalePrice"], axis = 1).to_numpy()-X_mu)/X_sigma
X = np.c_[np.ones(X.shape[0]), X]

step = 40
error_train, error_CV = learningCurve(X, y, X_CV, y_CV, lambda_ = lambda_optimal, step = step)
num_data = np.arange(1, y.size+1, step)

learning_curve = pd.DataFrame()
learning_curve["Data Quantity"] = num_data
learning_curve["Train Error"] = error_train
learning_curve["CV Error"] = error_CV

print(learning_curve)
plt.plot(num_data, error_train, label="Train"); plt.plot(num_data, error_CV, label = "CV"); plt.legend(); plt.xlabel("Num of Data"); plt.ylabel("Error"); plt.title("Learning Curve"); plt.show()

"""Final Model Train"""
"""
TRAINING THE MODEL ON THE WHOLE TRAIN SET - NO CV!!!
CV =  all_data.sample(frac=0.2)
y_CV = (CV["SalePrice"].to_numpy()-y_mu)/y_sigma
X_CV = (CV.drop(["SalePrice"], axis = 1).to_numpy()-X_mu)/X_sigma
X_CV = np.c_[np.ones(X_CV.shape[0]), X_CV]
data = all_data.drop(CV.index)
"""
train = all_data
y = (train["SalePrice"].to_numpy()-y_mu)/y_sigma
X = (train.drop(["SalePrice"], axis = 1).to_numpy()-X_mu)/X_sigma
X = np.c_[np.ones(X.shape[0]), X]

theta = trainModel(X, y, lambda_ = lambda_optimal)

"""Test data"""
raw_test = pd.read_csv('./data/raw_test.csv')
test = pd.read_csv('./data/test.csv')
Id = raw_test["Id"]

X_test = test.to_numpy()
X_test = (X_test - X_mu)/X_sigma
m,n = np.shape(X_test)
X_test = np.c_[np.ones(m), X_test]

y_predicted = predict(theta, X_test)*y_sigma + y_mu

solution = pd.DataFrame()
solution["Id"] = Id
solution["SalePrice"] = y_predicted
solution.to_csv("./linearRegression/solution.csv", sep=",", index=False)
