import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from linRegFuncsOld import *

"""Sorint Train & CV Data"""
train = pd.read_csv("./data/train.csv")
y_train = train["SalePrice"]
X_train = train.drop(["SalePrice", "Id"], axis = 1)
y = y_train.to_numpy()
X = X_train.to_numpy()
#X, mu, sigma = featureNormalisation(X)
#y, y_mu, y_sigma = featureNormalisation(y)
m,n = np.shape(X)
X = np.c_[np.ones(m), X]
y = np.reshape(y, (m,))


CV = pd.read_csv("./data/CV.csv")
y_CV = CV["SalePrice"]
X_CV = CV.drop(["SalePrice", "Id"], axis = 1)
y_CV = y_CV.to_numpy()
X_CV = X_CV.to_numpy()
#X_CV = (X_CV - mu)/sigma
#y_CV = (y_CV - y_mu)/y_sigma
m,n = np.shape(X_CV)
X_CV = np.c_[np.ones(m), X_CV]
y_CV = np.reshape(y_CV, (m,))

"""Basic Model Test"""
theta = trainModel(X, y)
p = predict(theta, X_CV)

"""CV Error VS Lambdas"""
lambdas = np.array([0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])

error_train, error_CV = validationCurve(X, y, X_CV, y_CV, lambdas)

lambda_errors = pd.DataFrame()
lambda_errors["Lambda"] = lambdas
lambda_errors["Train Error"] = error_train
lambda_errors["CV Error"] = error_CV

plt.semilogx(lambdas, error_train, label = "Train"); plt.semilogx(lambdas, error_CV, label = "CV"); plt.legend(); plt.xlabel("Lambdas"); plt.ylabel("Errors"); plt.title("Validation Curve"); plt.show()
print(lambda_errors)

"""Using Optimal Lambda To Output Thetas"""
np.random.seed()

lambda_optimal = 10**3
theta = trainModel(X, y, lambda_ = lambda_optimal)

p = predict(theta, X_CV)
print("Optimised model on CV error = ",np.mean( np.abs(p-y_CV)/y_CV))

"""Generating CV & Train Learning Curves"""
step = 50
error_train, error_CV = learningCurve(X, y, X_CV, y_CV, lambda_ = lambda_optimal, step = step)
num_data = np.arange(1, y.size+1, step)

learning_curve = pd.DataFrame()
learning_curve["Data Quantity"] = num_data
learning_curve["Train Error"] = error_train
learning_curve["CV Error"] = error_CV

print(learning_curve)
plt.plot(num_data, error_train, label="Train"); plt.plot(num_data, error_CV, label = "CV"); plt.legend(); plt.xlabel("Num of Data"); plt.ylabel("Error"); plt.title("Learning Curve"); plt.show()

"""Generating Test Results"""
test = pd.read_csv('./data/test.csv')
passengerId = test["Id"]
test = test.drop("Id", axis=1)

X_test = test.to_numpy()
X_test = (X_test - mu)/sigma
m,n = np.shape(X_test)
X_test = np.c_[np.ones(m), X_test]

y_predicted = predict(theta, X_test)*y_sigma + y_mu

solution = pd.DataFrame()
solution["Id"] = passengerId
solution["SalePrice"] = y_predicted
solution.to_csv("./linearRegression/solution.csv", sep=",", index=False)

