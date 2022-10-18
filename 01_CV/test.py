import glob

import torchvision.transforms
from PIL import Image
from model import *

imags_path = glob.glob("/Users/yingying/Desktop/picture/*")
print(imags_path)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
model = torch.load("./yy_model_6.pth")
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=False,
                                         transform=torchvision.transforms.ToTensor())
for p in imags_path:
    image = transform(Image.open(p))
    image = torch.reshape(image, (1, 3, 32, 32))
    model.eval()
    with torch.no_grad():
        output = model(image)
    print(p, train_set.classes[output.argmax(1).item()])


