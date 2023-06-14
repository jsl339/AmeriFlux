import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import xgboost as xgb
import os

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LearningCurveDisplay
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor


#Read CSV to pandas dataframe, header first row 
df_chunk=pd.read_csv("/Users/johnleland/Downloads/Condensed44+GCC.csv", header = 0, chunksize=10000, low_memory= False)
chunk_list = []  # append each chunk df here 
# Each chunk is in df format
for chunk in df_chunk: 
    
    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk)
    
# concat the list into dataframe 
df = pd.concat(chunk_list)

print("Dataframe Loaded")


X = df.drop(["FC50","Site"], axis = 1)
Y = df["FC50"]

params = {'subsample': 0.6, 'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.6}

# {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}

model = xgb.XGBRegressor(objective="reg:squarederror", random_state=13, n_estimators = 100, max_depth = 10,
                          learning_rate=0.1, subsample = 0.6, colsample_bytree = 0.6, colsample_bylevel = 0.5, early_stopping_rounds = 10)
overallmse = []
overallr2 = []
cv = LeaveOneGroupOut()
for i, (train, test) in enumerate(cv.split(X, Y, groups=df["Site"])):
    print("Fold Number:", i)
    start = time.time()
    x_train = X.iloc[train]
    y_train = Y.iloc[train]

    x_test = X.iloc[test]
    y_test = Y.iloc[test]

    evalset = [(x_train,y_train),(x_test,y_test)]
    # Model
    model.fit(x_train,y_train, eval_set = evalset, verbose = False)

    #Prediction
    predict2 = model.predict(x_test)

    #Model Assessment
    mse = mean_squared_error(y_test,predict2)
    rmse = np.sqrt(mse)
    score = model.score(x_test,y_test)
    overallmse.append(rmse)
    overallr2.append(score)
    end = time.time()
    elapsed = end - start
    print("Time Per Fold:", elapsed)
    print(f'RMSE: {rmse}, R2 Score: {score}')


    # Plots
    # retrieve performance metrics
    results = model.evals_result()
    # plot learning curves
    plt.plot(results['validation_0']['rmse'], label='train')
    plt.plot(results['validation_1']['rmse'], label='test')
    directory = "XGregression/FC50/"
    parent_dir = "/Users/johnleland/Desktop/Ameriflux_Data/"
    path = os.path.join(parent_dir, directory) 

    # os.mkdir(path)
    plt.savefig(path + "Learning_Curve"+str(i)+".png")
    plt.close()
    sorted_idx = model.feature_importances_.argsort()
    imp = model.feature_importances_[sorted_idx]
    ximp = x_test.columns[sorted_idx]
    fig1 = plt.barh(ximp[:10],imp[:10])
    plt.savefig(path + "Feature_Importances"+str(i)+".png")
    plt.close()
    d = {'Fold': [i], 'RMSE': [rmse], 'R2':[score]}
    saving = pd.DataFrame(data=d)
    saving.to_csv(path + "Metrics"+str(i)+".csv")
print("RMSE Average:",np.mean(overallmse), "RMSE Median:",np.median(overallmse),"R2 Average:",np.mean(overallr2), "R2 Median:", np.median(overallr2))
dd = {'Average RMSE': [np.mean(overallmse)], 'Average R2':[np.mean(overallr2)], 'Median RMSE':[np.median(overallr2)], 'Median R2':[np.median(overallr2)]}
ddsave = pd.DataFrame(data = dd)
ddsave.to_csv(path + "OverallMetrics.csv")