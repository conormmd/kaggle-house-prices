import numpy as np
import pandas as pd
from featureAnalysisFuncs import *

train = pd.read_csv('./data/raw_train.csv')
y_train = train["SalePrice"]
train = train.drop("SalePrice", axis = 1)
train_info = dtypeAnalysis(train)

dtype_num_tolerance = 1
nan_perc_tolerance = 0.15

dtype_num_flag, nan_perc_flag = flagProblematics(train_info)
train, train_info = removeProblematics(train, train_info)

numerical_categories = ["MSSubClass"]

train, feature_dict = featureSplittingTrain(train, train_info, numerical_categories)

train.to_csv("./data/train.csv", sep = ",", index=False)
y_train.to_csv("./data/y_train.csv", sep = ",", index=False)

test = pd.read_csv('./data/raw_test.csv')
test_info = dtypeAnalysis(test)

test = test.drop(dtype_num_flag, axis = 1)
test = test.drop(nan_perc_flag, axis = 1)

test = featureSplittingTest(test, feature_dict)

test.to_csv("./data/test.csv", sep = ",", index=False)