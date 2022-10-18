import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class my_sequential(nn.Module):
    def __init__(self):
        super(my_sequential, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, X):
        return self.model(X)


input = torch.ones((64, 3, 32, 32))
model = my_sequential()
print(model(input).shape)

writer = SummaryWriter("../0901")
writer.add_graph(model, input)
writer.close()
