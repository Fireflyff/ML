import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)
train_set = torchvision.datasets.CIFAR100(root="./dataset", train=True, download=False, transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=False, transform=dataset_transform)

print(len(train_set), len(test_set))
# img_train, target_train = train_set[0]
# img_test, target_test = test_set[0]
#
# print(len(train_set.classes), len(test_set.classes))
# print(train_set.classes[target_train], test_set.classes[target_test])
# img_train.show()
# img_test.show()
writer = SummaryWriter("../0830")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
writer.close()

