#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

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

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
#Define custom data class for data loader
class FluxData(Dataset):
    def __init__(self,x,y):
        #define shape, x_data, and y_data from input x and y torch tensors
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]

        self.x_data = x # size [n_samples, n_features]
        self.y_data = y # size [n_samples, 1]
       
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

#Define Dataset and loader

#Can change if using gpu for parallel computing)
# %%
class Linear(nn.Module):
    def __init__(self, n_features):
        super(Linear,self).__init__()
        #Define the Layers
        self.network=nn.Sequential(
        nn.Linear(n_features,100),
        nn.ReLU(),
        nn.Linear(100,40),
        nn.ReLU(),
        nn.Linear(40,10),
        nn.ReLU(),
        nn.Linear(10,1)
        )
        
    def forward(self,x):
        y=self.network(x)
        return y

for i,sitename in zip(iter,uni):
    ## data setup
    df_test = df[df['Site'] == sitename]
    df_train = df[df['Site'] != sitename]
    
    x_train = df_train.drop(df_train.filter(regex='FC|NEE|Site').columns, axis=1)
    y_train = df_train['FC50']

    x_test = df_test.drop(df_test.filter(regex='FC|NEE|Site').columns, axis=1)
    y_test = df_test['FC50']
    
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    
    x_train =torch.from_numpy(x_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_train = torch.Tensor(y_train.values).reshape(-1,1)
    y_test = torch.Tensor(y_test.values).reshape(-1,1)
    
    train_data = FluxData(x_train,y_train)
    train_loader = DataLoader(dataset=train_data,
                          batch_size=100,
                          shuffle=True,
                          num_workers=0)
    # Model 
    model = Linear(x_train.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr =0.00003, weight_decay = 0.001)
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        for j,(inputs, labels) in enumerate(train_loader):
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 99 or epoch == num_epochs-1:
                if (j+1) % 1000 == 0:
                    print(f'Fold: {i} Epoch: {epoch+1}/{num_epochs}, Step {j+1}| Loss = {loss.item():.3f}')
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        score = r2_score(y_pred,y_test)
        mse = mean_squared_error(y_pred,y_test)
        overallmse.append(mse)
        overallr2.append(score)
        print(f'Fold {i}:  Sitename:{sitename}  R2 Score: {score}   MSE: {mse}' )
    del model



# %%
