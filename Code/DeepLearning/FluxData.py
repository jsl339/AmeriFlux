# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
# %%
df = pd.read_csv('5BatchFIXED.csv')
# %%
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
df = df.dropna()

#Define x and y
df_test = df[df['Site'] == 'PR-xGU']
df_train = df[df['Site'] != 'PR-xGU']
    
x_train = df_train.drop(df_train.filter(regex='FC|NEE|Site').columns, axis=1)
y_train = df_train['FC50']

x_test = df_test.drop(df_test.filter(regex='FC|NEE|Site').columns, axis=1)
y_test = df_test['FC50']

# %%
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
import sklearn

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
#Convert x and y to tensors
y_train = torch.Tensor(y_train.values).reshape(-1,1)
y_test = torch.Tensor(y_test.values).reshape(-1,1)
#Scale data
x_scaler = sklearn.preprocessing.StandardScaler()
X_train = x_scaler.fit_transform(x_train)
X_test = x_scaler.transform(x_test)


X_train =torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))


#Define Dataset and loader
train_data = FluxData(X_train,y_train)
train_loader = DataLoader(dataset=train_data,
                          batch_size=100,
                          shuffle=True,
                          num_workers=0)#Can change if using gpu for parallel computing)

                          
#Define simple feed forward model
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

#%%
#model, loss, and optimizer
model = Linear(X_train.shape[1])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr =0.0005, weight_decay = 0.001)

#empty loss arrays to visualize training and test loss
train_loss = []
test_loss = []

#Training Loop (Very, Very slow, so only do 100 epochs until using HPC)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i,(inputs, labels) in enumerate(train_loader):
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 4 or epoch == num_epochs-1:
            if (i+1) % 500 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}| Loss = {loss.item():.3f}')
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            y_pred1 = model(X_train)
            y_pred2 = model(X_test)
        
        
            train_loss.append(loss_fn(y_pred1,y_train))
            test_loss.append(loss_fn(y_pred2,y_test))
# %%
with torch.no_grad():
    fig, ax = plt.subplots()

    ax.plot(train_loss, label="Training")
    ax.plot(test_loss, label="Test")
    ax.legend(loc=0); # upper left corner
    ax.set_xlabel('Epochs (x5)')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Loss Visualization')
    
    fig.savefig('plot.png')
# %%
with torch.no_grad():
    print(f'train r2: {sklearn.metrics.r2_score(model(X_train), y_train)} ')
    print(f'test r2: {sklearn.metrics.r2_score(model(X_test), y_test)}')

# %%
#torch.save(model.state_dict(),r"/home/jru34/savedmodel2")
