from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter("../yy_1_logs")
img = Image.open("/Users/yingying/Desktop/WX20220829-165204@2x.png")
print(img)
writer.add_image("origin", np.array(img), dataformats='HWC')

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("toTensor", img_tensor)
print(img_tensor)

print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 1)

print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0, 0, 0, 0], [1, 1, 1, 1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

writer.close()
