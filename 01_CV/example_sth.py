import torch

input = torch.tensor([[[1, 2, 3]]])
print(input.size())
input = input.unsqueeze(0)
print(input.size())
input = input.squeeze()
print(input.size())