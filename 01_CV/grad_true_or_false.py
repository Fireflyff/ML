import torch
from torch import nn


class YY_nn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, input):
        return self.linear(input)


# model = YY_nn()
# x = torch.rand(10, 10)
# print(x)
# output = model(x)
# print(output.requires_grad, output.grad_fn)
# with torch.no_grad():
#     output = model(x)
# print(output.requires_grad, output.grad_fn)


x = torch.tensor(1.1, requires_grad=True)
y = x ** 2
z = y + 5.2
z.backward()
print(x.grad)
print(y.grad)
print(x.is_leaf)
print(y.is_leaf)
print(x.grad_fn)
print(y.grad_fn)
