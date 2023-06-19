#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression,Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

def Partitionedvalidation(dataset, model, folds, Deg=0):
    x = dataset.drop(['FC50','Site'], axis = 1)
    y = dataset["FC50"]
    gkf = KFold(n_splits=folds)
    overallmse = []
    overallr2 = []
    print(f'Training on {folds} folds\n--------------------------------')
    for i, (train, test) in enumerate(gkf.split(x, y)):
        x_train = x.iloc[train]
        x_test= x.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        if Deg>0:
            poly = PolynomialFeatures(Deg)
            x_train = poly.fit_transform(x_train)
            x_test = poly.transform(x_test)
            
        lin = model().fit(x_train,y_train)
        y_pred = lin.predict(x_test)
        score = r2_score(y_pred,y_test)
        score2 = r2_score(lin.predict(x_train),y_train)
        mse = mean_squared_error(y_pred,y_test.values)
        overallmse.append(mse)
        overallr2.append(score)
        print(f'Fold {i+1}: R2 train score: {score2}  R2 test Score: {score}   MSE: {mse}')
    
    print(f'\nLinear Regression - Average MSE: {sum(overallmse)/len(overallmse)}, Average R2: {sum(overallr2)/len(overallr2)}')
    print(f'\nModel weights \n------------------------------')
    if Deg >0:
        featurenames = featurenames=poly.get_feature_names_out()
    else:
        featurenames=list(x_train.columns)
    for item in zip(featurenames,lin.coef_):
        print(item)
    overallmse.append(np.mean(overallmse))
    overallr2.append(np.mean(overallr2))
    metrics = {'MSE': overallmse, 'R2': overallr2}
    metricdf = pd.DataFrame(metrics)
    return metricdf
    

#%%
df = pd.read_csv(r"C:\Users\jeffu\Documents\Condensed44+GCC.csv")
dropcols = ['midday_gcc','midday_rcc','gcc_75','rcc_75','gcc_90','rcc_90','Rh','SW_DIF', 'SW_IN_1_1_1', 'SW_OUT']
df.drop(dropcols,axis=1,inplace=True)
#%%
df1 = Partitionedvalidation(df,LinearRegression,10)
df1.to_csv('LinearMetrics')
df2=Partitionedvalidation(df,LinearRegression,10,2)
df2.to_csv('QuadraticMetrics')
# %%
