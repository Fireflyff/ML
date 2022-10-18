import torch
from torch import nn


class YY_nn(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

    
model = YY_nn()
x = torch.tensor(1.0)
output = model(x)
print(output)
output = model(1.)
print(output)