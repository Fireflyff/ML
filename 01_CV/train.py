import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10("./dataset", True, transform=torchvision.transforms.ToTensor(),
                                          download=False)
test_data = torchvision.datasets.CIFAR10("./dataset", False, transform=torchvision.transforms.ToTensor(),
                                         download=False)
train_loader = DataLoader(train_data, 64)
test_loader = DataLoader(test_data, 64)

print(len(train_data))
print(len(test_data))

my_model = yy_model()

loss_fun = nn.CrossEntropyLoss()

optim = torch.optim.SGD(my_model.parameters(), lr=1e-2)

train_step = 0
test_step = 0

epoch = 7

writer = SummaryWriter("../0902")

for i in range(epoch):
    # train
    for data in train_loader:
        imgs, target = data
        output = my_model(imgs)
        loss = loss_fun(output, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_step = train_step + 1
        if train_step % 100 == 0:
            print("训练次数{}，loss: {}".format(train_step, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), train_step)

    # test
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, target = data
            output = my_model(imgs)
            loss = loss_fun(output, target)
            total_test_loss += loss
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy

        print(f"测试集上的总loss: {total_test_loss}")
    # writer.add_scalar("test_loss", total_test_loss, i)
    writer.add_scalar("test_acc", total_accuracy, i)
    # torch.save(my_model, f"yy_model_{i}.pth")

writer.close()


