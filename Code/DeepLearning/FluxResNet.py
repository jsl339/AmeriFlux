#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

class BasicBlock(nn.Module):
    
    def __init__(self, input):
        super(BasicBlock, self).__init__()
        self.Lin1 = nn.Linear(input,input)
        self.relu = nn.ReLU(inplace=True)
        self.Lin2 = nn.Linear(input, input)

    def forward(self, x):
        identity = x
        out = self.Lin1(x)
        out = self.relu(out)
        out = self.Lin2(out)
        out += identity
        out = self.relu(out)
        return out

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

print(f'CUDA available: {torch.cuda.is_available()}')
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
#Convert x and y to tensors
y_train = torch.Tensor(y_train.values).reshape(-1,1)
y_test = torch.Tensor(y_test.values).reshape(-1,1)
#Scale data
x_scaler = sklearn.preprocessing.StandardScaler()
X_train = x_scaler.fit_transform(x_train)
X_test = x_scaler.transform(x_test)


X_train =torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

if torch.cuda.is_available():
    X_train = X_train.cuda()
    X_test = X_test.cuda()
    y_train = y_train.cuda()
    y_test = y_test.cuda()

print(f'x_train is on {X_train.get_device()}')
#Define Dataset and loader
train_data = FluxData(X_train,y_train)
train_loader = DataLoader(dataset=train_data,
                          batch_size=100,
                          shuffle=True,
                          num_workers=0)
#Can change if using gpu for parallel computing)

block_1 = BasicBlock(60)
block_2 = BasicBlock(60)
block_3 = BasicBlock(40)
block_4 = BasicBlock(40)
block_5 = BasicBlock(20)
block_6 = BasicBlock(20)
block_7 = BasicBlock(10)
block_8 = BasicBlock(10)                       
#Define simple feed forward model
class Linear(nn.Module):
    def __init__(self, n_features):
        super(Linear,self).__init__()
        #Define the Layers
        self.network=nn.Sequential(
        nn.Linear(n_features,60),
        nn.ReLU(),
        block_1,
        block_2,
        nn.Linear(60,40),
        nn.ReLU(),
        block_3,
        block_4,
        nn.Linear(40,20),
        nn.ReLU(),
        block_5,
        block_6,
        nn.Linear(20,10),
        nn.ReLU(),
        block_7,
        block_8,
        nn.Linear(10,1)
        )
        
    def forward(self,x):
        y=self.network(x)
        return y

#%%
#model, loss, and optimizer
model = Linear(X_train.shape[1])
if torch.cuda.is_available():
    model = model.cuda()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr =0.00003, weight_decay = 0.001)

#empty loss arrays to visualize training and test loss
train_loss = []
test_loss = []
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
#Training Loop (Very, Very slow, so only do 100 epochs until using HPC)
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for i,(inputs, labels) in enumerate(train_loader):
        #if torch.cuda.is_available():
            #inputs = inputs.cuda()
            #labels = labels.cuda()
        y_pred = model(inputs)
        if epoch == 0 and i ==0:
            print(f'y_pred is on device {y_pred.get_device()}')
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 4 or epoch == num_epochs-1:
            if (i+1) % 500 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}| Loss = {loss.item():.3f}')
    #if epoch % 5 == 0:
  
#Training Loop (Very, Very slow, so only do 100 epochs until using HPC)
    model.eval()
    with torch.no_grad():
        y_pred1 = model(X_train)
        y_pred2 = model(X_test)
    
    
        train_loss.append(loss_fn(y_pred1,y_train).item())
        test_loss.append(loss_fn(y_pred2,y_test).item())
# %%
with torch.no_grad():
    fig, ax = plt.subplots()

    ax.plot(train_loss, label="Training")
    ax.plot(test_loss, label="Test")
    ax.legend(loc=0); # upper left corner
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Loss Visualization')
    
    fig.savefig('plot.png')
# %%
with torch.no_grad():
    print(f'train r2: {sklearn.metrics.r2_score(model(X_train).cpu(), y_train.cpu())} ')
    print(f'test r2: {sklearn.metrics.r2_score(model(X_test).cpu(), y_test.cpu())}')

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
torch.save(model,r"/home/jru34/models/savedDeepModel")
# %%
