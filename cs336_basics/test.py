import torch
import torch.nn as nn

x = torch.Tensor([[
    [1, 2, 3, 4],
    [4, 5, 6, 6],
    [7, 8, 9, 9]
],
[
    [3, 2, 1, 9],
    [6, 4, 5, 9],
    [9, 1, 7, 9]
]])

print(x.shape)
x = x.view(2, 3, 2, 2)
print(x)
x = x.transpose(1, 2)
print(x)