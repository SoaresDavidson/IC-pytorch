# arquivo inutil apenas para testar ideias idiotas
import torch
import torch.nn as nn
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

x = torch.arange(6*14*14, dtype=float).resize_(6,14,14)
print(x)
print(x.sum(dim=(0)))
print(x.mean(dim = 0))


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
