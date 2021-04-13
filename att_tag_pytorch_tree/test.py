import torch
import torch.nn as nn

# x= torch.Tensor( [ 1,2,3,4])

# y1 = torch.nn.functional.softmax(x).max(0)[1] #对每一列进行softmax
# print(y1)
l = [1,2]
for i in l:
	i = i + 1
print(l)

# x = torch.randn((1,4),dtype=torch.float32,requires_grad=False)
# y = x ** 2
# z = y * 4
# print(x)
# print(y)
# print(z)
# for i in x[0]:
# 	print(i)
# loss1 = z.mean()
# loss2 = z.sum()
# print(loss1,loss2)
# loss1.backward()
# print(loss1.item())    # 这个代码执行正常，但是执行完中间变量都free了，所以下一个出现了问题
#print(loss1,loss2)
#loss2.backward()    # 这时会引发错误


# # at beginning of the script
# device = torch.device("cuda:0" if
# torch.cuda.is_available() else "cpu")

# ...

# # then whenever you get a new Tensor or Module
# # this won't copy if they are already on the desired device
# input = data.to(device)
# model = MyModule(...).to(device)