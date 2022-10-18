import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True, drop_last=True)

writer = SummaryWriter("../data_loader")
step = 0
for data in test_loader:
    imgs, target = data
    writer.add_images("batch_data_3", imgs, step)
    step += 1
writer.close()
