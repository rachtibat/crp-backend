import torch
import torch.nn as nn

class FashionLeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = nn.Conv2d(1, 6, 5)
        self.l2 = nn.AvgPool2d(2)
        self.l3 = nn.Conv2d(6, 16, 5)
        self.l4 = nn.AvgPool2d(2)
        self.l5 = nn.Flatten()
        self.l6 = nn.Linear(256, 120)
        self.l7 = nn.Linear(120, 84)
        self.l8 = nn.Linear(84, 10)

        self.activation = nn.ReLU()

    def forward(self, X):

        X = self.l1(X)
        X = self.activation(X)
        X = self.l2(X)
        X = self.l3(X)
        X = self.activation(X)
        X = self.l4(X)
        X = self.l5(X)
        X = self.l6(X)
        X = self.activation(X)
        X = self.l7(X)
        X = self.activation(X)
        X = self.l8(X)

        return X