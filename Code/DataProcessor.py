import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import dask.dataframe as dd

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LearningCurveDisplay
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeRegressor


df_chunk=pd.read_csv("/Users/johnleland/Desktop/Ameriflux_Data/Dataset/Index/44Datasets.csv", header = 0, chunksize=100000, low_memory= False)
chunk_list = []  # append each chunk df here 
# Each chunk is in df format
for chunk in df_chunk: 
    
    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk)
    
# concat the list into dataframe 
df = pd.concat(chunk_list)

#data = df[df.Ustar >= df.U5]
#df["FC5"] = data["FC"]

data = df[df.Ustar >= df.U50]
df["FC50"] = data["FC"]

#data = df[df.Ustar >= df.U95]
#df["FC95"] = data["FC"]

# L2 = ['YEAR', 'MONTH', 'DAY', 'DOY', 'HOUR', 'MINUTE', 'TS_1_1_1', 'TS_1_2_1', 'PPFD', 'PPFD_OUT','Tair',
      #'HourCos', 'HourSin', 'DOYCos', 'DOYSin', 'VPD', 'Rg',"Site","FC50"]
L = ["YEAR","MONTH","DAY","DOY","HOUR","MINUTE","FC50","Tair","SWC_1_1_1","SWC_1_2_1","SWC_1_3_1","SWC_2_1_1","PPFD","PPFD.1","PPFD.2","PPFD_OUT","TS_1_1_1","TS_1_2_1","TS_1_3_1",
      "TS_1_4_1","TS_1_5_1","TS_1_6_1","TS_1_7_1","TS_1_8_1","TS_1_9_1","Site","HourCos","VPD","DOYCos","Tair.1","Tair.2", "SW_DIF"]

df1 = df[L]

#df1 = df1.dropna().compute()

def outlier(data):
    data.dropna(inplace=True)
    x = sorted(data)
    q1, q3= np.percentile(x,[25,75])
    outside = 6
    iqr = q3 - q1
    lower_bound = q1 -(outside * iqr) 
    upper_bound = q3 +(outside * iqr) 
    return lower_bound,upper_bound


x = outlier(df1["FC50"])
x[0]


d = df1[df1["FC50"] <= x[1]]

d = d[d["FC50"] >= x[0]]

dpreds = d.drop(["FC50"], axis = 1)
dresponse = d["FC50"]


dpreds = dpreds.interpolate()

dat = dpreds.join(dresponse)

df2 = dat.dropna()
