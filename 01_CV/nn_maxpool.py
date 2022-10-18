
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)
test = DataLoader(dataset, batch_size=64)



class my_maxpool(nn.Module):
    def __init__(self):
        super(my_maxpool, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # output = self.maxpool1(X)
        output_1 = self.relu(X)
        output_2 = self.sigmoid(X)
        return output_1, output_2

model = my_maxpool()
writer = SummaryWriter("../0901")
step = 0

for data in test:
    imgs, _ = data
    # writer.add_images("maxpool0", imgs, step)
    output_1, output_2 = model(imgs)
    writer.add_images("maxpool2", output_1, step)
    writer.add_images("maxpool3", output_2, step)
    step = step + 1

writer.close()
