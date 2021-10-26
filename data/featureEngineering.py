import numpy as np
import pandas as pd
from featureAnalysisFuncs import *

train = pd.read_csv('./data/raw_train.csv')
train_info = dtypeAnalysis(train)

dtype_num_tolerance = 1
nan_perc_tolerance = 0.15

dtype_num_flag, nan_perc_flag = flagProblematics(train_info)
train, train_info = removeProblematics(train, train_info, dtype_num_flag, nan_perc_flag)

numerical_categories = ["MSSubClass"]

train, feature_dict = featureSplittingTrain(train, train_info, numerical_categories)

feature_info, feature_zero_flag = featureAnalysis(train, zero_perc_threshold=0.98)

train = train.drop(feature_zero_flag, axis = 1)


"""CV Set"""
data_CV = train.sample(frac=0.2)
data_CV.to_csv("./data/CV.csv",  sep=",", index=False)

"""Train Set"""
train = train.drop(data_CV.index)
train.to_csv("./data/train.csv", sep=",", index=False)


test = pd.read_csv('./data/raw_test.csv')
test_info = dtypeAnalysis(test)

test = test.drop(dtype_num_flag, axis = 1)
test = test.drop(nan_perc_flag, axis = 1)

test = featureSplittingTest(test, feature_dict)
test = test.drop(feature_zero_flag, axis = 1)

test.to_csv("./data/test.csv", sep = ",", index=False)