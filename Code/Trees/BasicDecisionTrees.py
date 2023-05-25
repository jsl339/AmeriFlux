import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Read CSV to pandas dataframe, header first row 
df=pd.read_csv('/Users/johnleland/Desktop/Ameriflux_Data/5BatchFIXED.csv', header = 0, low_memory=False)
print("------------------------------Read CSV-----------------------------------")

data = df[df.Ustar >= df.U5]
df["FC5"] = data["FC"]

data = df[df.Ustar >= df.U50]
df["FC50"] = data["FC"]

data = df[df.Ustar >= df.U95]
df["FC95"] = data["FC"]

drop_cols = ['DateTime','TIMESTAMP', 'TIMESTAMP_START','TIMESTAMP_END','Ulevel',"aggregationMode",'season']

df.drop(drop_cols,axis=1,inplace=True)

df = df.dropna()

#Use pd 'category type' to make a list of integers for each site name
groups = df['Site'].astype('category')
groups1 = groups.cat.codes


print("------------------------------Data Cleaned-----------------------------------")

uni = list(np.unique(df["Site"]))
overallmse = []
overallr2 = []
iter = [1,2,3,4,5]

model1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=100, random_state=1)
model2 = DecisionTreeRegressor(max_depth=40)
print("------------------------------Model Created-----------------------------------")

for i,sitename in zip(iter,uni):
    print("Beginning iteration:",i)
    ## data setup
    df_test = df[df['Site'] == sitename]
    df_train = df[df['Site'] != sitename]
    
    x_train = df_train.drop(df_train.filter(regex='FC|NEE|Site').columns, axis=1)
    y_train = df_train['FC50']

    x_test = df_test.drop(df_test.filter(regex='FC|NEE|Site').columns, axis=1)
    y_test = df_test['FC50']
    # Model
    #model1.fit(x_train,y_train)
    model2.fit(x_train,y_train)

    #Prediction
    #predict1 = model1.predict(x_test)
    predict2 = model2.predict(x_test)

    #Model Assessment

    score1 = r2_score(y_test, predict2)
    mse1 = mean_squared_error(y_test,predict2)
    overallmse.append(mse1)
    overallr2.append(score1)
    print(f'Fold {i}:  R2 Score: {score1}',f'Fold {i}:  MSE: {mse1}' )

print("R2 Average:",np.mean(overallr2),"MSE Average:",np.mean(overallmse))
