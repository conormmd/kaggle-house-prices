import numpy as np
import pandas as pd
from collections import Counter

def dtypeAnalysis(data):
    """Initialising Lists to hold Test results"""
    col_headers = data.columns.tolist()
    col_non_nan = []
    col_nans = []
    col_num_dtype = []
    col_strs = []
    col_ints = []
    col_floats = []
    m, n = data.shape

    for i in col_headers:
        """Initialising a temporary Test Column"""
        temp_col = data[i]

        """Testing nans, dropping nan rows for next Tests"""
        nans = list(np.where(data[i].isna())[0])
        col_nans.append(len(nans))
        temp_col = temp_col.drop(temp_col.index[nans])
        m_col = temp_col.size
        col_non_nan.append(m_col/m)

        """Testing dtypes (sans nans), counting type of each"""
        data_list = temp_col.tolist()
        data_types = [str(type(i).__name__) for i in data_list]
        dtype_count = Counter(data_types)
        col_num_dtype.append(len(dtype_count))
        col_strs.append(dtype_count["str"])
        col_ints.append(dtype_count["int"])
        col_floats.append(dtype_count["float"])


    """Initialising a DataFrame to store Test results"""
    col_info = pd.DataFrame()
    col_info["Headers"] = col_headers
    col_info["Non Nan %"] = col_non_nan
    col_info["Nans"] = col_nans
    col_info["Num Dtypes"] = col_num_dtype
    col_info["Strings"] = col_strs
    col_info["Ints"] = col_ints
    col_info["Floats"] = col_floats

    return col_info

def flagProblematics(data_info, dtype_num_tolerance = 1, nan_perc_tolerance = 0.15):
    dtype_num_flag = []
    nan_perc_flag = []

    m, n = data_info.shape

    for i in range(m):
        temp_data = data_info.loc[i]
        header = temp_data["Headers"]
        non_nan_perc = temp_data["Non Nan %"]
        nan_num = temp_data["Nans"]
        dtype_num = temp_data["Num Dtypes"]
           
        if dtype_num > dtype_num_tolerance:
            dtype_num_flag.append(header)
            continue
        else:
            pass

        if non_nan_perc < nan_perc_tolerance:
            nan_perc_flag.append(header)
            continue
        else:
            pass
    return dtype_num_flag, nan_perc_flag

def removeProblematics(data, data_info, dtype_num_flag, nan_perc_flag, dtype_num_tolerance = 1, nan_perc_tolerance = 0.15):

    data = data.drop(dtype_num_flag, axis = 1)
    data = data.drop(nan_perc_flag, axis = 1)

    data_info = data_info.drop( data_info.index[data_info["Non Nan %"] < nan_perc_tolerance].tolist() )
    data_info = data_info.drop( data_info.index[data_info["Num Dtypes"] > dtype_num_tolerance].tolist() )
    return data, data_info

def categoricalFeaturesTrain(data, header):
    nans = len(list(np.where(data[header].isna())[0]))
    classes = data[header].unique()
    if nans != 0:
        data[header] = data[header].fillna(0)
        classes = [i for i in classes if str(i) != 'nan']
    classes = sorted(classes)
    if len(classes) <= 1:
        pass
    if len(classes) == 2:
        i = classes[0]
        data[str(header)+"_"+str(i)] = (data[header] == i).astype(int)
        features = [i]
    else:
        for i in classes:
            data[str(header)+"_"+str(i)] = (data[header] == i).astype(int)
        features = classes
    data = data.drop(header, axis = 1)
    return data, features

def numericalFeaturesTrain(data, header):
    data[header] = data[header].fillna(data[header].mean())
    features = 0
    return data, features

def featureSplittingTrain(data, data_info, numerical_categories):

    feature_dict = {}

    m, n = data_info.shape
    data_info = data_info.reset_index(drop = True)

    headers = data.columns.tolist()

    for i in range(m):
        temp_data = data_info.loc[i]

        header = temp_data["Headers"]
        nan_num = temp_data["Nans"]
        str_num = temp_data["Strings"]
        int_num = temp_data["Ints"]
        float_num = temp_data["Floats"]

        if header in numerical_categories:
            data, features = categoricalFeaturesTrain(data, header)
            feature_dict[header] = features
            continue
        
        if str_num > 0:
            data, features = categoricalFeaturesTrain(data, header)
            feature_dict[header] = features
            continue
        
        if int_num or float_num > 0:
            data, features = numericalFeaturesTrain(data, header)
            feature_dict[header] = features
            continue
    return data, feature_dict

def numericalFeaturesTest(data, header):
    data[header] = data[header].fillna(data[header].mean())
    return data

def categoricalFeaturesTest(data, header, classes):
    data[header] = data[header].fillna(0)
    classes = [i for i in classes if str(i) != 'nan']
    if len(classes) <= 1:
        pass
    if len(classes) == 2:
        i = classes[0]
        data[str(header)+"_"+str(i)] = (data[header] == i).astype(int)
    else:
        for i in classes:
            data[str(header)+"_"+str(i)] = (data[header] == i).astype(int)
    data = data.drop(header, axis = 1)
    return data

def featureSplittingTest(data, feature_dict):
    headers = data.columns.tolist()
    for i in headers:
        if type(feature_dict[i]) == int:
            data = numericalFeaturesTest(data, i)
            continue
        else:
            data = categoricalFeaturesTest(data, i, feature_dict[i])
            continue
    return data

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
