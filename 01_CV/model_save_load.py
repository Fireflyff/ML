import torch
import torchvision.models

# 方法一，保存网络结构和参数
# vgg16 = torchvision.models.vgg16(False)
# torch.save(vgg16, "vgg16_method1.pth")
# print(torch.load("vgg16_method1.pth"))


# 方法二，将网络结构，保存成字典 (省空间)

# vgg16 = torchvision.models.vgg16(False)
# torch.save(vgg16.state_dict(), "vgg16_method2.pth")
MODEL = torch.load("vgg16_method2.pth")
print(MODEL)
