# 定义LSTM的参数含义：（input_size, hidden_size, num_layers）
# 定义输入张量的参数含义：（sequence_length, batch_size, input_size）
# 定义隐藏层初始张量和细胞初始状态张量的参数含义：
# （num_layers * num_directions, batch_size, hidden_size）

import torch.nn as nn
import torch
rnn = nn.LSTM(5, 6, 2)
input = torch.randn(1, 3, 5)
h0 = torch.randn(2, 3, 6)
c0 = torch.randn(2, 3, 6)
output, (hn, cn) = rnn(input, (h0, c0))
print(output.shape, h0.shape, c0.shape)
print("#" * 100)
print(output)
print("*" * 100)
print(h0)
print("%" * 100)
print(c0)