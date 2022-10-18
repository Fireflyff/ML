import torch.optim
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader


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


dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)
train = DataLoader(dataset, 64)
network = my_sequential()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(network.parameters(), lr=0.01)

for epoch in range(100):
    iter_loss = 0
    for data in train:
        imgs, target = data
        output = network(imgs)
        result_loss = loss(output, target)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        iter_loss += result_loss
    print(iter_loss)