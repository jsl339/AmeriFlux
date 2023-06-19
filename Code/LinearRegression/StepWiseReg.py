#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


#Data Setup
df = pd.read_csv(r"C:\Users\jeffu\Documents\43Datasets+GCC.csv")
#%%

#data = df[df.Ustar >= df.U5]
#df["FC5"] = data["FC"]

data = df[df.Ustar >= df.U50]
df["FC50"] = data["FC"]

del data

#data = df[df.Ustar >= df.U95]
#df["FC95"] = data["FC"]

drop_cols = ['Unnamed: 0',
 'YEAR',
 'DateTime',
 'MONTH',
 'DAY',
 'DOY',
 'HOUR',
 'MINUTE',
 'TIMESTAMP',
 'TIMESTAMP_START',
 'TIMESTAMP_END',
 'H2O_1_1_2',
 'H2O_1_1_3',
 'H2O_1_2_2',
 'H2O_1_2_3',
 'H2O_1_3_2',
 'H2O_1_3_3',
 'H2O_1_4_2',
 'H2O_1_4_3',
 'TS_1_3_1',
 'TS_1_4_1',
 'TS_1_5_1',
 'TS_1_6_1',
 'TS_1_7_1',
 'TS_1_8_1',
 'TS_1_9_1',
 'PPFD.1',
 'PPFD.2',
 'PPFD.3',
 'PPFD_BC_IN_2_1_1',
 'PPFD_BC_IN_3_1_1',
 'Tair.1',
 'Tair.2',
 'Tair.3',
 'Tair.4',
 'VPD_PI',
 'NEE_PI',
 'Ustar',
 'SW_IN_1_1_2',
 'SW_IN_1_1_3',
 'SWC_1_2_1',
 'SWC_1_3_1',
 'SWC_1_4_1',
 'SWC_1_5_1',
 'SWC_1_6_1',
 'SWC_1_7_1',
 'SWC_1_8_1',
 'SWC_2_1_1',
 'SWC_2_2_1',
 'SWC_2_3_1',
 'SWC_2_4_1',
 'SWC_2_5_1',
 'SWC_2_6_1',
 'SWC_2_7_1',
 'SWC_2_8_1',
 'SWC_3_1_1',
 'SWC_3_2_1',
 'SWC_3_3_1',
 'SWC_3_4_1',
 'SWC_3_5_1',
 'SWC_3_6_1',
 'SWC_3_7_1',
 'SWC_3_8_1',
 'SWC_4_1_1',
 'SWC_4_2_1',
 'SWC_4_3_1',
 'SWC_4_4_1',
 'SWC_4_5_1',
 'SWC_4_6_1',
 'SWC_4_7_1',
 'SWC_4_8_1',
 'SWC_5_1_1',
 'SWC_5_2_1',
 'SWC_5_3_1',
 'SWC_5_4_1',
 'SWC_5_5_1',
 'SWC_5_6_1',
 'SWC_5_7_1',
 'SWC_5_8_1',
 'NEE',
 'aggregationMode',
 'season',
 'uStar',
 'U5',
 'U50',
 'U95',
 'FC']
df.drop(drop_cols, axis = 1, inplace = True)
#%%
df.dropna(inplace=True)
uni = list(np.unique(df["Site"]))[:10]
overallmse = []
overallr2 = []
iter = np.arange(len(uni))+1
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

#%%
print('Stepwise Regression with Sklearn LinearRegression')

#Forward Stepwise Regression
sfs = SequentialFeatureSelector(lin,n_features_to_select= 10, direction='forward')
sfs.fit(x,y)
x_cols = np.array(sfs.get_feature_names_out())
X = df[x_cols]
print(f'Top 5 Features: {x_cols}')
print(f'Top Scores:')
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
