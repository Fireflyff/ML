import torch.utils.data
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)
data = DataLoader(dataset, batch_size=64)

class YY_nn(nn.Module):
    def __init__(self):
        super(YY_nn, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, X):
        x = self.conv1(X)
        return x

model = YY_nn()
print(model)

step = 0
writer = SummaryWriter("../0831")
for i_data in data:
    imgs, targets = i_data
    output = model(imgs)
    # torch.Size([64, 3, 32, 32])
    print(imgs.shape)
    # torch.Size([64, 6, 30, 30])
    print(output.shape)
    writer.add_images("origin", imgs, step)

    output = torch.reshape(output, (-1, 3, 30, 30))
    print(output.shape)
    writer.add_images("conv_after", output, step)

    step = step + 1