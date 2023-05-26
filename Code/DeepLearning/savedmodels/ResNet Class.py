import torch
import torch.nn as nn

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