
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as T

import numpy as np

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


classes = {
            0: "T - shirt / top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }


def load_data(path):

    return np.load(path)


if __name__ == "__main__":
        
    model = FashionLeNet() 
    model.load_state_dict(torch.load("FashionLeNet.p"))
    model.eval()

    softmax = torch.nn.Softmax(dim=-1)

    test_set = torchvision.datasets.FashionMNIST(
            root='data/FashionMNIST',
            train=False,
            download=True,
            transform=T.Compose([
                T.ToTensor()
            ])
        )

    sample, label = test_set[0]

    pred = model(sample.unsqueeze(0))

    max_arg = torch.argmax(softmax(pred), dim=-1)

    print("Predicted class ", classes[max_arg])




    
