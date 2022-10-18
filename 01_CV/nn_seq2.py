import torch
from torch import nn
from torch.nn import Embedding, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class my_sequential(nn.Module):
    def __init__(self):
        super(my_sequential, self).__init__()
        self.model = Sequential(
            Embedding(30, 6),
            Linear(6, 3)
        )
    def forward(self, X):
        return self.model(X)


input = torch.tensor([3, 6, 9, 23])
model = my_sequential()
print(model(input).shape)

writer = SummaryWriter("../0914")
writer.add_graph(model, input)
writer.close()
