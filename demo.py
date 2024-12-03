import torch

a = [(0, 1), (1, 2), (2, 5), (3, 6)]
a = torch.tensor(a)
# 或者使用T进行转置
transposed_tensor2 = a.T

print("原始张量：\n", a)
print("转置后的张量（使用T）：\n", transposed_tensor2)
