from math import cos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from linRegFuncs import *

all_data = pd.read_csv("./data/all_train.csv")
all_y = all_data["SalePrice"].to_numpy()
all_X = all_data.drop(["Id", "SalePrice"], axis = 1).to_numpy()
all_X, X_mu, X_sigma = featureNormalisation(all_X)
all_y, y_mu, y_sigma = featureNormalisation(all_y)
k = 10

CV =  all_data.sample(frac=1/k)
X_CV = (CV.drop(["Id", "SalePrice"], axis = 1).to_numpy()-X_mu)/X_sigma
data = all_data.drop(CV.index)

train = data.sample(frac=1/(k-1))
