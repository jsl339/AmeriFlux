#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


#Data Setup
df = pd.read_csv('5BatchFIXED.csv', header = 0, low_memory=False)

data = df[df.Ustar >= df.U5]
df["FC5"] = data["FC"]

data = df[df.Ustar >= df.U50]
df["FC50"] = data["FC"]

data = df[df.Ustar >= df.U95]
df["FC95"] = data["FC"]


uni = list(np.unique(df["Site"]))
overallmse = []
overallr2 = []
iter = [1,2,3,4,5]
drop_cols = ['DateTime','TIMESTAMP', 'TIMESTAMP_START','TIMESTAMP_END','Ulevel',"aggregationMode",'season','FC5','FC95']
df.drop(drop_cols, axis = 1, inplace = True)
df.dropna(inplace=True)
#%%
print('Scores using all features:')
for i,sitename in zip(iter,uni):
    ## data setup
    df_test = df[df['Site'] == sitename]
    df_train = df[df['Site'] != sitename]
    
    x_train = df_train.drop(df_train.filter(regex='FC|NEE|Site').columns, axis=1)
    y_train = df_train['FC50']

    x_test = df_test.drop(df_test.filter(regex='FC|NEE|Site').columns, axis=1)
    y_test = df_test['FC50']
   
    
    # Model
    model = LinearRegression().fit(x_train,y_train)
    y_pred = model.predict(x_test)
    score = r2_score(y_pred,y_test)
    score2 = r2_score(model.predict(x_train),y_train)
    mse = mean_squared_error(y_pred,y_test.values)
    overallmse.append(mse)
    overallr2.append(score)
    print(f'Fold {i}:  Sitename:{sitename} R2 train score: {score2}  R2 test Score: {score}   MSE: {mse}' )

#%%
from sklearn.feature_selection import SequentialFeatureSelector
#Set desired number of features
lin = LinearRegression()
x = df.drop(df.filter(regex='FC|NEE|Site').columns, axis=1)
y = df['FC50']
tot_features = x.shape[1]
num_features = np.arange(start=1,stop=tot_features//2,step=5)
#%%
print('Stepwise Regression with Sklearn LinearRegression')
for num in num_features:
#Forward Stepwise Regression
    sfs = SequentialFeatureSelector(lin,n_features_to_select= num, direction='forward')
    sfs.fit(x,y)
    x_cols = np.array(sfs.get_feature_names_out())
    X = df[x_cols]
    print(f'Top {num} Features: {x_cols}')
    print(f'Top {num} Scores:')
    for i,sitename in zip(iter,uni):
        ## data setup
        df_test = df[df['Site'] == sitename]
        df_train = df[df['Site'] != sitename]
        
        x_train = df_train[x_cols]
        y_train = df_train['FC50']

        x_test = df_test[x_cols]
        y_test = df_test['FC50']
        
        #x_scaler = StandardScaler()
        #x_train = x_scaler.fit_transform(x_train)
        #x_test = x_scaler.transform(x_test)
        
        # Model 
        model = LinearRegression().fit(x_train,y_train)
        y_pred = model.predict(x_test)
        score = r2_score(y_pred,y_test)
        score2 = r2_score(model.predict(x_train),y_train)
        mse = mean_squared_error(y_pred,y_test.values)
        overallmse.append(mse)
        overallr2.append(score)
        print(f'Fold {i}:  Sitename:{sitename} R2 Score: {score}   MSE: {mse}' )



# %%
