import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import xgboost as xgb
import os

from sklearn import tree
from sklearn.model_selection import LearningCurveDisplay
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error,r2_score


#Read CSV to pandas dataframe, header first row 
df=pd.read_csv('/Users/johnleland/Desktop/Ameriflux_Data/5BatchFIXED.csv', header = 0, low_memory=False)
print("-------------------------------Read CSV-------------------------------------")

#data = df[df.Ustar >= df.U5]
#df["FC5"] = data["FC"]

data = df[df.Ustar >= df.U50]
df["FC50"] = data["FC"]

#data = df[df.Ustar >= df.U95]
#df["FC95"] = data["FC"]


drop_cols = ['DateTime','TIMESTAMP', 'TIMESTAMP_START','TIMESTAMP_END','Ulevel',"aggregationMode",'season']


df.drop(drop_cols,axis=1,inplace=True)

df = df.dropna()

#Use pd 'category type' to make a list of integers for each site name
groups = df['Site'].astype('category')
groups1 = groups.cat.codes


print("------------------------------Data Cleaned-----------------------------------")

uni = list(np.unique(df["Site"]))
iter = [1,2,3,4,5]
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=13, max_depth = 50)

print("------------------------------Model Created-----------------------------------")

overallmse = []
overallr2 = []
scorestrain=[]
scorescv=[]

for i,sitename in zip(iter,uni):
    start = time.time()
    print("Beginning iteration:",i)
    ## data setup
    df_test = df[df['Site'] == sitename]
    df_train = df[df['Site'] != sitename]
    
    x_train = df_train.drop(df_train.filter(regex='FC|NEE|Site|uStar|U|Rg').columns, axis=1)
    y_train = df_train['FC50']
    x_test = df_test.drop(df_test.filter(regex='FC|NEE|Site|uStar|U|Rg').columns, axis=1)
    y_test = df_test['FC50']

    evalset = [(x_train,y_train),(x_test,y_test)]
    # Model
    #model1.fit(x_train,y_train)
    model.fit(x_train,y_train, eval_set = evalset, verbose = False)

    #Prediction
    #predict1 = model1.predict(x_test)
    predict2 = model.predict(x_test)

    #Model Assessment
    mse = mean_squared_error(y_test,predict2)
    score = r2_score(y_pred= predict2, y_true=y_test)
    overallmse.append(mse)
    overallr2.append(score)
    end = time.time()
    elapsed = end - start
    print("Time Per Fold:", elapsed)
    print(f'Fold {i}:  MSE: {mse}, R2 Score: {score}')
    

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
    d = {'Fold': [i], 'MSE': [mse], 'R2':[score]}
    saving = pd.DataFrame(data=d)
    saving.to_csv(path + "Metrics"+str(i)+".csv")

print("MSE Average:",np.mean(overallmse), np.mean(overallr2))
dd = {'Average MSE': [np.mean(overallmse)], 'Average R2':[np.mean(overallr2)]}
ddsave = pd.DataFrame(data = dd)
ddsave.to_csv(path + "OverallMetrics.csv")


