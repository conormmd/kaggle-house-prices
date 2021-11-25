import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from featureAnalysisFuncs import *

train = pd.read_csv('./data/raw_train.csv')
train_info = dtypeAnalysis(train)

dtype_num_tolerance = 1
nan_perc_tolerance = 0.15

dtype_num_flag, nan_perc_flag = flagProblematics(train_info)
train, train_info = removeProblematics(train, train_info, dtype_num_flag, nan_perc_flag)

numerical_categories = ["MSSubClass"]

train, feature_dict = featureSplittingTrain(train, train_info, numerical_categories)

data = train

def featureAnalysis(data, zero_perc_threshold = 0.9):
    col_headers = data.columns.tolist()
    m,n = data.shape
    zeros_perc = []

    for i in col_headers:
            """Initialising a temporary Test Column"""
            temp_col = data[i].tolist()
            """Testing zeros"""
            zeros = temp_col.count(0)
            zeros_perc.append(zeros/m)

    feature_info = pd.DataFrame()
    feature_info["Headers"] = col_headers
    feature_info["0 %"] = zeros_perc

    zero_perc_threshold = 0.95
    feature_zero_flag = []

    m,n = feature_info.shape

    for i in range(m):
        temp_data = feature_info.loc[i]
        header = temp_data["Headers"]
        zeros_perc = temp_data["0 %"]

        if zeros_perc > zero_perc_threshold:
            feature_zero_flag.append(header)

    return feature_info, feature_zero_flag

feature_info, feature_zero_flag = featureAnalysis(train, zero_perc_threshold = 0.9)
