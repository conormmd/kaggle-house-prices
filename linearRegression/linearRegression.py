import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from linRegFuncs import *

train = pd.read_csv("./data/train.csv")
y_train = pd.read_csv("./data/y_train.csv")

train = train.drop("Id", axis = 1)

y = y_train.to_numpy()
X = train.to_numpy()

m,n = np.shape(X)
X = np.c_[np.ones(m), X]

theta = trainModel(X, y)

print(theta)
