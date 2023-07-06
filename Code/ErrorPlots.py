#%%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#%%
df = pd.read_csv(r"C:\Users\jeffu\Documents\FC50dataVer2.csv")
#%%
x_scaler = StandardScaler()
X = df.drop(['Site','FC50'],axis=1)
X = x_scaler.fit_transform(X)
X = torch.Tensor(X)
y = df['FC50']
#Define simple feed forward model
class Linear(nn.Module):
    def __init__(self, n_features):
        super(Linear,self).__init__()
        #Define the Layers
        self.lin1 = nn.Linear(n_features,256)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(256,128)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(128,64)
        self.act3=nn.ReLU()
        self.out = nn.Linear(64,1)
        self.drop = nn.Dropout1d(p=0.6)
        '''
        nn.init.xavier_normal_(self.lin1.weight)
        nn.init.xavier_normal_(self.lin2.weight)
        nn.init.xavier_normal_(self.lin3.weight)
        nn.init.xavier_normal_(self.out.weight)'''
    def forward(self,x):
        y=self.lin1(x)
        y=self.act1(y)
        y=self.lin2(y)
        y=self.act2(y)
        #y=self.drop(y)
        y=self.lin3(y)
        y=self.act3(y)
        y=self.out(y)
        return y
#%%
model = Linear(X.shape[1])

model.load_state_dict(torch.load(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Summer 2023 Research\Models\FC50Deep.pt"))

y_pred = model(X)

df['Predicted'] = y_pred.detach()

d = df.loc[:,['DOY','FC50', 'Predicted']]
grouped = d.groupby('DOY')

# %%
y_avg = []
pred_avg = []

for doy, group in grouped:
    y_avg.append(np.mean(group.FC50))
    pred_avg.append(np.mean(group.Predicted))

y_avg = np.array(y_avg)
pred_avg = np.array(pred_avg) 
# %%
plt.plot (y_avg, '.', label = 'FC50')
plt.plot(pred_avg, 'r.', label= 'Prediction')
plt.title('Average Daily Error')
plt.xlabel('Day of Year')
plt.legend()
# %%
y_avg = np.array(y_avg)
pred_avg = np.array(pred_avg)
err = pred_avg - y_avg
plt.plot (err, '.', label = 'net error')
plt.title('Average Daily Error')
plt.xlabel('Day of Year')
plt.legend()

# %%
plt.plot (np.abs(err), '.', label = 'Absolute error')
plt.title('Average Daily Error')
plt.xlabel('Day of Year')
plt.legend()   
# %%
