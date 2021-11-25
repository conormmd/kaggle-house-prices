import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from linRegFuncs import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("./data/all_train.csv")

print(dataset.shape)