import pandas as pd
import numpy as np
import os
import time
import glob

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut

#Read CSV to pandas dataframe, header first row 
df_chunk=pd.read_csv("/Users/johnleland/Downloads/LE50Data.csv", header = 0, chunksize=10000, low_memory= False)
chunk_list = []  # append each chunk df here 
# Each chunk is in df format
for chunk in df_chunk: 
    
    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk)
    
# concat the list into dataframe 
df = pd.concat(chunk_list)
print("-------------------------------Read CSV-------------------------------------")

X = df.drop(["LE50","Site"], axis = 1)
Y = df["LE50"]

# model = DecisionTreeRegressor(max_depth=10,random_state= 13)
print("------------------------------Model Created-----------------------------------")

cv = KFold(n_splits=10,shuffle=True,random_state= 13)


overallmse = []
overallr2 = []
model = DecisionTreeRegressor(max_depth=10,random_state= 13)
for i, (train, test) in enumerate(cv.split(X, Y)):
    print("Fold Number:", i)
    start = time.time()
    x_train = X.iloc[train]
    y_train = Y.iloc[train]

    x_test = X.iloc[test]
    y_test = Y.iloc[test]
    # Model
    model.fit(x_train,y_train)

    #Prediction
    predict2 = model.predict(x_test)

    #Model Assessment
    mse = mean_squared_error(y_test,predict2)
    score = model.score(x_test,y_test)
    overallmse.append(mse)
    overallr2.append(score)
    end = time.time()
    elapsed = end - start
    print("Time Per Fold:", elapsed)
    print(f'MSE: {mse}, R2 Score: {score}')
    directory = "BasicDecisionTrees/LE50"
    parent_dir = "/Users/johnleland/Desktop/Ameriflux_Data/"
    path = os.path.join(parent_dir, directory) 

    d = {'Fold': [i], 'MSE': [mse], 'R2':[score]}
    saving = pd.DataFrame(data=d)
    saving.to_csv(path + "Metrics_Kfold"+str(i)+".csv")
print("MSE Average:",np.mean(overallmse), "MSE Median:",np.median(overallmse),"R2 Average:",np.mean(overallr2), "R2 Median:", np.median(overallr2))
dd = {'Average MSE': [np.mean(overallmse)], 'Average R2':[np.mean(overallr2)], 'Median MSE':[np.median(overallmse)], 'Median R2':[np.median(overallr2)]}
ddsave = pd.DataFrame(data = dd)
ddsave.to_csv(path + "/OverallMetrics_KFold.csv")

def collection(path_name):
    files = glob.glob((path_name))
    print(files)
    pdm = {}
    pdm = [pd.read_csv(f) for f in files]
    merged = pd.concat(pdm)
    merged = merged.sort_values(by=["Fold"])
    merged.to_csv(path + "/Results.csv")
    return merged

collection(path)
print("All Done")