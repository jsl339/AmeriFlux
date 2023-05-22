import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

dataset = pd.read_csv("/Users/johnleland/Desktop/Ameriflux_Data/5BatchSite.csv",header=1)

def Partitionedvalidation(dataset, folds):
    col_list =  (list(dataset["Site"]))
    x = dataset[dataset.columns.drop("FC")]
    y = dataset["FC"]
    gkf = GroupKFold(n_splits=folds)
    for train, test in gkf.split(x, y, groups=col_list):
        print(train, test)
    return(train,test)
    
def GroupValidation(dataset, folds):
    col_list =  np.array(dataset["Site"])
    x = dataset[dataset.columns.drop("FC")]
    y = dataset["FC"]
    gss = GroupShuffleSplit(n_splits= folds, test_size= 0.2,random_state=13)
    split = gss.split(x, y, col_list)
    train_inds, test_inds = next(split)
    train = dataset.iloc[train_inds]
    test = dataset.iloc[test_inds]
    return(train,test)
