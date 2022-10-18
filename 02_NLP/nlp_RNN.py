import torch
from torch import nn
# rnn = nn.GRU(100, 10, num_layers=7)
rnn = nn.RNN(100, 10, num_layers=7)
X = torch.rand(9, 3, 100)
Y, stae_new = rnn(X, None)
print(Y.shape, len(stae_new), stae_new.shape)
print("#"*100)
print(Y, stae_new)
print("*"*100)
# print(rnn._parameters.items())
for key, values in rnn._parameters.items():
    print(key, values.shape)
    # print(data)
    # print(data[0], data[1].shape)