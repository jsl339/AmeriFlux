import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error,r2_score


#Read CSV to pandas dataframe, header first row 
df_chunk=pd.read_csv('/Users/johnleland/Downloads/43Datasets+GCC.csv', header = 0, chunksize=100000, low_memory= False)
chunk_list = []  # append each chunk df here 
# Each chunk is in df format
for chunk in df_chunk: 
    
    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk)
    
# concat the list into dataframe 
df = pd.concat(chunk_list)

print("-------------------------------Read CSV-------------------------------------")


data = df[df.Ustar >= df.U50]
df["FC50"] = data["FC"]

#data = df[df.Ustar >= df.U95]
#df["FC95"] = data["FC"]

L2 = ['YEAR', 'MONTH', 'DAY', 'DOY', 'HOUR', 'MINUTE', 'TS_1_1_1', 'TS_1_2_1', 'PPFD', 'PPFD_OUT','Tair',
      'VPD_PI','HourCos', 'HourSin', 'DOYCos', 'DOYSin', 'VPD', 'Rg',"Site","FC50",'smooth_gcc_90', 'smooth_rcc_90',
       'smooth_gcc_50', 'smooth_rcc_50', 'smooth_gcc_75', 'smooth_rcc_75']
df1 = df[L2]
print(df1.columns)

#df1 = df1.dropna().compute()

df1.dropna(inplace=True)



#Use pd 'category type' to make a list of integers for each site name
groups = df['Site'].astype('category')
groups1 = groups.cat.codes


print("------------------------------Data Cleaned-----------------------------------")

uni = list(np.unique(df["Site"]))

iter = range(1,len(uni))
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=13, n_estimators=2, learning_rate=0.05)

print("------------------------------Model Created-----------------------------------")

overallmse = []
overallr2 = []

for i,sitename in zip(iter,uni):
    start = time.time()
    print("Beginning iteration:",i)
    ## data setup
    df_test = df[df['Site'] == sitename]
    df_train = df[df['Site'] != sitename]

    x_train = df_train.drop(df_train.filter(regex='FC|Site').columns, axis=1)
    y_train = df_train['FC50']
    x_test = df_test.drop(df_test.filter(regex='FC|Site').columns, axis=1)
    y_test = df_test['FC50']


    #evalset = [(x_train,y_train),(x_test,y_test)]
    # Model
    #model.fit(x_train,y_train, eval_set = evalset, verbose = False)
    model.fit(x_train,y_train, verbose = False)

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
"""
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
"""

print("MSE Average:",np.mean(overallmse), np.mean(overallr2))
#dd = {'Average MSE': [np.mean(overallmse)], 'Average R2':[np.mean(overallr2)]}
#ddsave = pd.DataFrame(data = dd)
#ddsave.to_csv(path + "OverallMetrics.csv")
