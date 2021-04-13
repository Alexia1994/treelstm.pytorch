import torch
import torch.nn as nn
torch.manual_seed(48)

net = nn.Linear(100, 8000)
for i in net.named_parameters():
	print(i)
#####
# 维度越大初始化的向量越小

