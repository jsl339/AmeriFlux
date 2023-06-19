#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

print('Preparing Data \n--------------------------------')
#df = pd.read_csv('~/data/Condensed44+GCC.csv')
df = pd.read_csv(r"C:\Users\jeffu\Documents\Condensed44+GCC.csv", header = 0, low_memory=False)
dropcols = ['midday_gcc','midday_rcc','gcc_75','rcc_75','gcc_90','rcc_90','Rh','SW_DIF', 'SW_IN_1_1_1', 'SW_OUT']
df.drop(dropcols,axis=1,inplace=True)
uni = list(np.unique(df["Site"]))

iter = np.arange(len(uni))+1

#%%

overallmse = []
overallr2 = []
print('Starting Linear Regression \n--------------------------------------')
for i,sitename in zip(iter,uni):
    ## data setup
    df_test = df[df['Site'] == sitename]
    df_train = df[df['Site'] != sitename]
    
    x_train = df_train.drop(['FC50','Site'], axis=1)
    y_train = df_train['FC50']

    x_test = df_test.drop(['FC50','Site'],axis=1)
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
print(f'Linear Regression - Average MSE: {sum(overallmse)/len(overallmse)}, Average R2: {sum(overallr2)/len(overallr2)}')
print(f'\nModel weights \n------------------------------')
featurenames=list(x_train.columns)
for item in zip(featurenames,model.coef_):
    print(item)
    
overallmse.append(np.mean(overallmse))
overallr2.append(np.mean(overallr2))
  
metrics = {'MSE': overallmse, 'R2': overallr2}
metricdf = pd.DataFrame(metrics)
metricdf.to_csv('LinearLOGOmetrics.csv')
# %%
overallmse = []
overallr2 = []
print('Starting Polynomial Regression\n------------------------------')
poly = PolynomialFeatures(2)
for i,sitename in zip(iter,uni):
    ## data setup
    df_test = df[df['Site'] == sitename]
    df_train = df[df['Site'] != sitename]
    
    x_train = df_train.drop(['FC50','Site'], axis=1)
    y_train = df_train['FC50']

    x_test = df_test.drop(['FC50','Site'],axis=1)
    y_test = df_test['FC50']
    
    
    x_train = poly.fit_transform(x_train)
    x_test = poly.transform(x_test)
    
    
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
print(f'Degree 2 Polynomial- Average MSE: {sum(overallmse)/len(overallmse)}, Average R2: {sum(overallr2)/len(overallr2)}')
print(f'Model weights \n------------------------------')
featurenames=poly.get_feature_names_out()
for item in zip(featurenames,model.coef_):
    print(item)
    
overallmse.append(np.mean(overallmse))
overallr2.append(np.mean(overallr2))
  
metrics = {'MSE': overallmse, 'R2': overallr2}
metricdf = pd.DataFrame(metrics)
metricdf.to_csv('QuadLOGOmetrics.csv')

# %%
