import numpy as np
import pandas as pd

data = pd.read_csv('./data/raw_train.csv')

"""Dealing with MSSubClass (Dwelling Type)"""
Header = "MSSubClass"
Classes = data[Header].unique()
for i in Classes:
    data[str(Header)+"_"+str(i)] = (data[Header]==i).astype(int)
data.drop(Header, axis = 1)

"""Dealing with MSZoning (Zoning Type)"""
Header = "MSZoning"
Classes = data[Header].unique()
for i in Classes:
    data[str(Header)+"_"+str(i)] = (data[Header]==i).astype(int)
data.drop(Header, axis = 1)

"""Dealing with LotFrontage (Street Connected to Property)"""
Header = "LotFrontage"
data[Header] = data[Header].fillna(data[Header].mean())

"""Dealing with LotArea"""
Header = "Area"

"""Dealing with Street"""
Header = "Street"
data["Street_Pave"] = (data[Header]=="Pave").astype(int)
data.drop(Header, axis = 1)

"""Dealing with Alley"""
Header = "Alley"
Classes = ["Pave", "Grvl"]
for i in Classes:
    data[str(Header)+"_"+str(i)] = (data[Header]==i).astype(int)
data.drop(Header, axis = 1)

"""Dealing with LotShape"""
Header = "LotShape"
Classes = data[Header].unique()
for i in Classes:
    data[str(Header)+"_"+str(i)] = (data[Header]==i).astype(int)
data.drop(Header, axis = 1)

"""Dealing with LotShape"""
Header = "LandContour"
Classes = data[Header].unique()
for i in Classes:
    data[str(Header)+"_"+str(i)] = (data[Header]==i).astype(int)
data.drop(Header, axis = 1)

"""Dealing with Utilities (Too few data)"""
Header = "Utilities"
data.drop(Header, axis = 1)

"""Dealing with LotConfig"""
Header = "LotConfig"
Classes = data[Header].unique()
for i in Classes:
    data[str(Header)+"_"+str(i)] = (data[Header]==i).astype(int)
data.drop(Header, axis = 1)

data.to_csv("./data/test.csv", sep = ",", index=False)

