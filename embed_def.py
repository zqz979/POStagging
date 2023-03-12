import torch.nn.functional as F
from torch import nn

class Embedder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,in_features,num_classes):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,embedding_dim)
        self.fc1=nn.Linear(in_features=in_features*embedding_dim,out_features=32)
        self.fc2=nn.Linear(in_features=32,out_features=16)
        self.fc3=nn.Linear(in_features=16,out_features=num_classes)
    def forward(self,x):
        pred=self.embed(x).view((x.shape[0],-1))
        pred=F.relu(self.fc1(pred))
        pred=F.relu(self.fc2(pred))
        pred=self.fc3(pred)
        return pred