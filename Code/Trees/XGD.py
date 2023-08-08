import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import random
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LearningCurveDisplay, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, RepeatedKFold
from sklearn.tree import DecisionTreeRegressor

from scipy.stats import boxcox, shapiro, kstest
from scipy.stats import t, sem

#Read CSV to pandas dataframe, header first row 
df_chunk=pd.read_csv("/Users/johnleland/Desktop/LE50DataVer_NEW.csv", header = 0, chunksize=10000, low_memory= False)

chunk_list = []  # append each chunk df here 
# Each chunk is in df format
for chunk in df_chunk: 
    
    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk)
    
# concat the list into dataframe 
df = pd.concat(chunk_list)

#df = df[df.LE50 >= 0 ]

#df = df[df.PPFD >= 50]

print("Dataframe Loaded")

X = df.drop(["LE50","Site", "prcp2week"], axis = 1)
Y = df["LE50"] # log/ box-cox transform?
# Xy = xgb.QuantileDMatrix(X, Y)
""""

Y,j = boxcox(Y)

Y = pd.DataFrame((Y))
"""
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=13, tree_method = "hist",n_estimators = 2000, early_stopping_rounds = 50, 
                        max_depth=6, learning_rate = 0.15, min_child_weight=1, subsample=0.6, colsample_bynode = 0.6) 
                        # "This is the best model for LE50 LOGO
#model = xgb.XGBRegressor(objective="reg:squarederror", random_state=13, tree_method = "hist",n_estimators = 2500, early_stopping_rounds = 20, 
 #                        learning_rate=0.025, max_depth=14, subsample=0.7,colsample_bynode =0.7)
                         
#model = xgb.XGBRegressor(objective="reg:squarederror", random_state=13, tree_method = "hist",n_estimators = 2000, early_stopping_rounds = 50, 
 #                        colsample_bynode =0.45, learning_rate=0.05, max_depth=10, min_child_weight=1, subsample=0.5) # FC50 model

overallmse = []
overallr2 = []
# cv = LeaveOneGroupOut()
cv = KFold(n_splits=10, shuffle=True, random_state=13)
for i, (train, test) in enumerate(cv.split(X, Y, groups=df["Site"])):
    print("Fold Number:", i)
    start = time.time()
    x_train = X.iloc[train]
    y_train = Y.iloc[train]

    x_test = X.iloc[test]
    y_test = Y.iloc[test]

    evalset = [(x_train,y_train),(x_test,y_test)]
    # Model
    model.fit(x_train,y_train, eval_set = evalset, verbose = True)

    #Prediction
    predict = model.predict(x_test)
    
    #Model Assessment
    mse = mean_squared_error(y_test,predict)
    score = r2_score(y_test,predict)
    overallmse.append(mse)
    overallr2.append(score)
    end = time.time()
    elapsed = end - start

    print("Time Per Fold:", elapsed)
    print(f'RMSE: {mse}, R2 Score: {score}')

    residual = y_test - predict

    sample_std = []
    for j in range(500):
        yk = random.choices(residual.tolist(), k = len(residual))
        avg = np.std(yk)
        sample_std.append(avg)

    mu = np.mean(sample_std)
    
    # Plots
    # retrieve performance metrics
    results = model.evals_result()
    # plot learning curves
    plt.plot(results['validation_0']['rmse'], label='train')
    plt.plot(results['validation_1']['rmse'], label='test')
    # model.best_iteration
    plt.legend()
    directory = "XGregression/FC50/LOGO/FC50"
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

    p = pd.Series(predict, index = y_test.index)
    l = pd.Series(predict-(2*mu), index = y_test.index) # 95 CI
    u = pd.Series((2*mu)+predict, index = y_test.index) # 95 CI
    dif = pd.concat([p,y_test,x_test["DOY"],l,u],axis=1)

    sq_error = pd.Series((y_test - predict), index = y_test.index)
    mse_partial = pd.concat([y_test, sq_error], axis = 1)
    mse_partial.columns = ["TrueValue", "SquaredError"]
    df_sort = mse_partial.sort_values(["TrueValue"])
    df_sort.reset_index(drop=True,inplace=True)

    plt.plot(df_sort.TrueValue,df_sort.SquaredError)
    plt.title('FC Error')
    plt.xlabel('FC50')
    plt.ylabel('Error')
    plt.savefig(path + "Squared_Error_Partitions"+str(i)+".png")
    plt.close()

    dif.columns = ["Prediction", "Test","DOY", "Lower","Upper"]
    y_avg = dif.groupby("DOY")["Test"].mean()
    pred_avg = dif.groupby("DOY")["Prediction"].mean()
    low_avg = dif.groupby("DOY")["Lower"].mean()
    up_avg = dif.groupby("DOY")["Upper"].mean()

    low_avg = low_avg.rolling(7).mean()
    up_avg = up_avg.rolling(7).mean()

    """plt.plot (y_avg, '.', label = 'FC50')
    plt.plot(pred_avg, 'b-', label= 'Prediction')
    #plt.plot(low_avg, 'r-', label= 'Lower Bound')
    #plt.plot(up_avg, 'r-', label= 'Upper Bound')
    plt.fill_between(y_avg.index, low_avg, up_avg, alpha=0.2)
    plt.title('Average Daily Error')
    plt.xlabel('Day of Year')
    plt.legend()
    plt.savefig(path + "Time_Series_Error_"+str(i)+".png")
    plt.close()"""

    ## new fig 

print("MSE Average:",np.mean(overallmse), "MSE Median:",np.median(overallmse),"R2 Average:",np.mean(overallr2), "R2 Median:", np.median(overallr2))
"""dd = {'Average MSE': [np.mean(overallmse)], 'Average R2':[np.mean(overallr2)], 'Median MSE':[np.median(overallmse)], 'Median R2':[np.median(overallr2)]}
ddsave = pd.DataFrame(data = dd)
ddsave.to_csv(path + "OverallMetrics_KFold.csv")"""