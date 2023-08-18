#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import os
from time import time

print('Preparing Data\n---------------------------')
df = df=pd.read_csv('~/data/LE50DataVer4.csv')
#df = pd.read_csv(r"C:\Users\jeffu\Documents\Condensed44+GCC.csv")

#%%
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
#Define simple feed forward model
class Linear(nn.Module):
    def __init__(self, n_features):
        super(Linear,self).__init__()
        #Define the Layers
        self.network=nn.Sequential(
        nn.Linear(n_features,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,1)
        )
        
    def forward(self,x):
        y=self.network(x)
        return y
#model, loss, and optimizer
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='Deep10checkpointLE.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def train_model(mod,num_epochs):
    start =time()
    print('Starting Training \n ----------------------------------------')
    model = mod
    train_loss = []
    test_loss = []
    train_r2 = []
    test_r2 = []
    
    early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in range(num_epochs):
        if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                print(f'Epoch {epoch+1} \n---------------------')
        model.train()
        for j,(inputs, labels) in enumerate(train_loader):
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                if (j+1) % 10 == 0:
                    print(f'Step {j+1}| Loss = {loss.item():.3f}')
    #if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            y_pred1 = model(X_train)
            y_pred2 = model(X_test)
        
        
            train_loss.append(mean_squared_error(y_pred=y_pred1.cpu(),y_true=y_train.cpu()))
            test_loss.append(mean_squared_error(y_pred=y_pred2.cpu(),y_true=y_test.cpu()))
            train_r2.append(r2_score(y_true=y_train.cpu(),y_pred=y_pred1.cpu()))
            test_r2.append(r2_score(y_true=y_test.cpu(),y_pred=y_pred2.cpu()))
        
        valid_loss = test_loss[-1]
        early_stopping(valid_loss,model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    model.load_state_dict(torch.load('Deep10checkpointLE.pt'))
    end = time()
    print(f'Training Complete, {epoch} epochs: Time Elapsed: {(end-start)//60} minutes, {(end-start)%60} seconds')
    return model, train_loss, test_loss, train_r2, test_r2
       
overallmse = []
overallr2 = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

cv = KFold(10,shuffle=True, random_state=13)

X = df.drop(['Site','LE50'], axis=1)
y = df['LE50']
#%%
for i,(train,test) in enumerate(cv.split(X,y)):
    ## data setup
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    y_train = y.iloc[train]
    y_test = y.iloc[test]
   
    #Convert x and y to tensors
    y_train = torch.Tensor(y_train.values).reshape(-1,1)
    y_test = torch.Tensor(y_test.values).reshape(-1,1)
    #Scale data
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    X_train =torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    
    model = Linear(X_train.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr =0.0003, weight_decay = 0)
    
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        X_test = X_test.cuda()
        y_train = y_train.cuda()
        y_test = y_test.cuda()
        model= model.cuda()
        
    train_data = FluxData(X_train,y_train)
    train_loader = DataLoader(dataset=train_data,
                            batch_size=128,
                            shuffle=True,
                            num_workers=0,
                            )
    #Can change if using gpu for parallel computing)
    #Training Loop (Very, Very slow, so only do 100 epochs until using HPC)
    
    

    model, train_loss, test_loss, train_r2, test_r2 = train_model(model,1000)
    
    with torch.no_grad():
        y_pred = model(X_test)
        score = r2_score(y_pred=y_pred.cpu(),y_true=y_test.cpu())
        score2 = r2_score(y_pred=model(X_train).cpu(),y_true=y_train.cpu())
        mse = mean_squared_error(y_pred=y_pred.cpu(),y_true=y_test.cpu())
        overallmse.append(mse)
        overallr2.append(score)
    print(f'\nFold {i}:\n----------\nR2 train score: {score2}\nR2 test Score: {score}\nMSE: {mse}' )
print(f'Average MSE: {np.mean(overallmse)}, Average R2: {np.mean(overallr2)}')
# %%
with torch.no_grad():
    fig,axs = plt.subplots(2,2)

    axs[0][0].plot(train_loss, label= 'Training Loss')
    axs[0][0].legend()
    axs[0][1].plot(test_loss, 'r', label = 'Test Loss'  )
    axs[0][1].legend()


    axs[1][0].plot(train_r2, label = 'Train R2')
    axs[1][0].legend()
    axs[1][1].plot(test_r2, 'r', label = 'Test R2')
    axs[1][1].legend()

    fig.savefig('Deeper10LE.png')
    overallmse.append(np.mean(overallmse))
    overallr2.append(np.mean(overallr2))
  
metrics = {'MSE': overallmse, 'R2': overallr2}
metricdf = pd.DataFrame(metrics)
metricdf.to_csv('LE_swin_filtered_10fold.csv')

# %%
