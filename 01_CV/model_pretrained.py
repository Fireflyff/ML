import torchvision.models
import ssl

from torch.nn import Linear

ssl._create_default_https_context = ssl._create_unverified_context
ssl.create_default_context()

model_false = torchvision.models.vgg16(pretrained=False)
model_true = torchvision.models.vgg16(pretrained=True)
model_true.classifier[0] = Linear(100, 10)
model_true.classifier.add_module("yy_layer", Linear(1000, 64))
print(model_false)
print(model_true)