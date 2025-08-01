import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# x = torch.randn(4,3)
# print(x.size())
# print(x.reshape(-1,2,2,1))
# print(x)

# x = torch.randn(3,4)
# y = torch.randn(4,3)
# print(x)
# print(y)
# y = torch.transpose(y,0 ,1)
# print(x + x)

"concatenar por eixo"
# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

"broadcasting mechanism"
# a = torch.arange(12).reshape((3, 4))
# b = torch.arange(4).reshape((1, 4))
# print(a, b)
# print(a+b)

# A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
# B = A.clone()  # Assign a copy of A to B by allocating new memory
# print(A, A + B)
# print(id(A) == id(B))

# x = torch.arange(12).reshape(3, 2, 2)
# print(x)
# print(x.sum(axis=0))
# print(x.sum(axis=1))

# x = torch.rand(2,3)
# print(x)
# print(x/x.sum(axis=0))

# x = torch.arange(4.0)
# # Can also create x = torch.arange(4.0, requires_grad=True)
# x.requires_grad_(True)
# print(x.grad)  # The gradient is None by default
# y = 2 * torch.dot(x, x)
# print(y)

# print(y.backward())
# print(x.grad)

# x.grad.zero_()  # Reset the gradient
# y = x.sum()
# y.backward()
# print(x.grad)

# x = torch.arange(24,dtype=float).reshape(3,2,2,2)
# print(x)
# print(x.mean(dim=1))


# inputDim = 1        # takes variable 'x' 
# outputDim = 1       # takes variable 'y'
# learningRate = 0.01 
# epochs = 100
# device = "cuda"
# # create dummy data for training
# x_values = [i for i in range(11)]
# x_train = np.array(x_values, dtype=np.float32)
# # print(x_train)
# x_train = x_train.reshape(-1, 1)
# # print(x_train)

# y_values = [2*i + 1 for i in x_values]
# y_train = np.array(y_values, dtype=np.float32)
# y_train = y_train.reshape(-1, 1)

# class linearRegression(torch.nn.Module):
#     def __init__(self, inputSize, outputSize):
#         super().__init__()
#         self.linear = torch.nn.Linear(inputSize, outputSize)

#     def forward(self, x):
#         out = self.linear(x)
#         return out

# model = linearRegression(inputDim, outputDim).to(device=device)
# print(model.parameters())
# criterion = torch.nn.MSELoss() 
# optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# for epoch in range(epochs):
#     # Converting inputs and labels to Variable
#     inputs = torch.from_numpy(x_train).cuda()
#     labels = torch.from_numpy(y_train).cuda()

#     # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
#     optimizer.zero_grad()

#     # get output from the model, given the inputs
#     outputs = model(inputs)

#     # get loss for the predicted output
#     loss = criterion(outputs, labels)
#     print(loss)
#     # get gradients w.r.t to parameters
#     loss.backward()

#     # update parameters
#     optimizer.step()

#     print('epoch {}, loss {}'.format(epoch, loss.item()))


# with torch.no_grad(): # we don't need gradients in the testing phase
#     predicted = model(torch.from_numpy(x_train).to(device=device))
#     print(predicted)

# plt.clf()
# plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
# plt.plot(x_train, predicted.cpu(), '--', label='Predictions', alpha=0.5)
# plt.legend(loc='best')
# plt.show()

# arquivo inutil apenas para testar ideias idiotas

##como funciona o keepdim=True
# x = torch.arange(2*2*2*2).resize_(2,2,2,2)
# print(x)
# y = x.sum((0,1,2),keepdim=True)
# print(y)
# print(y.size())

#teste de elemete-wise operations
# x = torch.arange(2*2).resize_(2,2)
# y = torch.arange(2*2).resize_(2,2)
# print(x.mul_(y))
# print(x)
# print(y)

# x = torch.randn(10,84)
# print(x)

# x = torch.arange(6*14*14, dtype=float).resize_(6,14,14)
# print(x)
# print(x.sum(dim=(0)))
# print(x.mean(dim = 0))


# import torch
# import torch.nn.functional as F

# # Input: batch_size=64, channels=20, H=W=14
# input_tensor = torch.randn(64, 20, 14, 14)

# # 2D kernel: shape [14, 14]
# kernel_2d = torch.randn(14, 14)

# # Expand to apply same kernel across all channels
# kernel_stack = kernel_2d.expand(20, 1, 14, 14)  # [out_channels, in_channels/groups, kH, kW]

# # Apply depthwise convolution (1 kernel per channel)
# output = F.conv2d(input_tensor, kernel_stack, groups=20)

# print(output.shape)  # Should be [64, 20, 1, 1] if no padding/stride

# class BinWeights(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         ctx.save_for_backward(x)
#         return torch.sign(x)
    
#     @staticmethod
#     def backward(ctx, grad_output): # type: ignore #1 back 2 back
#         print(grad_output)
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input.ge(1)] = 0
#         grad_input[input.le(-1)] = 0
#         return grad_input
    
# x = torch.randn(2,2,2,2, requires_grad=True)
# print(x)

# y = BinWeights.apply(x)
# print(y)
# loss = y.sum()  # or any scalar function of y
# loss.backward()

# print(x.grad)
# # print(y)

# x = torch.arange(16).resize_(2,2,4)
# print(x, x.size())
# y = torch.unsqueeze(x, 0)
# print(y, y.size())
# z = torch.unsqueeze(x, 1)
# print(z, z.size())

# net = nn.Sequential(nn.LazyLinear(8),
#                     nn.ReLU(),
#                     nn.LazyLinear(1))

# X = torch.rand(size=(2, 4))
# print(net(X).shape)
# print(net[0].state_dict())

# net = nn.Sequential(nn.Linear(2,1),
#                     )
# print(net[0].weight)

# def xnor_popcount(a: int, b: int) -> int:
#     return (~(a ^ b) & 7).bit_count() 
# print(xnor_popcount(2,3))

# import torch


def my_hook(grad):
    print("In hook, grad argument:", grad)


# v = torch.tensor([0., 0., 0.], requires_grad=True)
# lr = 0.01
# # simulate a simple SGD update
# h = v.register_post_accumulate_grad_hook(my_hook)
# v.backward(torch.tensor([1., 2., 3.]))
# print(v)
# print(v.grad)

# h.remove()  # removes the hook

# x = torch.rand(2)
# print(x)
# net = nn.Sequential(nn.Linear(2,1),
#                     nn.Linear(1,1)
#                     )
# print(net[0].weight)
# print(net[1].weight)

# net[1].weight.register_post_accumulate_grad_hook(my_hook)

# y = net(x)
# print(y)
# y.backward(torch.Tensor([1]))

x = torch.arange(16, dtype=float).resize_(2,2, 2,2)
print(x)
print(x.mean(1), x.mean(1).shape)
print(x.mean(1).expand_as(x))
print(x-(x.mean(1, keepdim=True).expand_as(x)))