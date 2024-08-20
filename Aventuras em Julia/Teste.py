import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self,n_hidden,n_visible,n_classes,learning_weight = 0.01):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible).to(device))
        self.U = nn.Parameter(torch.rand(n_hidden,n_classes).to(device))
        self.b = nn.Parameter(torch.zeros(n_visible).to(device))
        self.c = nn.Parameter(torch.zeros(n_hidden).to(device))
        self.d = nn.Parameter(torch.zeros(n_classes).to(device))
        self.leaning_rate = learning_weight

    def forward(self, x, y):
        x = x.to(device)
        y = y.to(device)
        h = F.sigmoid(self.c + np.matmul(self.W,x.T) + np.matmul(self.U,y.T))
        return h
    def train_rbm(self,X,Y):
        n_samples = X.shape[0]
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando {device}')