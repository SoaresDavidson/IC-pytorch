
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
class BinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class Binarize(nn.Module):
    def __init__(self):
        super(Binarize, self).__init__()

    def forward(self, x):
        return BinFunction.apply(x)
    
class BinConvParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        ctx.save_for_backward(weights)
        mean_weight = weights - weights.mean(1, keepdim=True).expand_as(weights)

        
        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .mean((1,2,3), keepdim=True)\
                .expand_as(weights)

        return clamped_weight.sign().mul(alpha)

    @staticmethod
    def backward(ctx, grad_output): #type: ignore
        weights, = ctx.saved_tensors
        saidas = weights[0].nelement() #num de saídas
        dim = weights.size() #dimensões

        alpha = weights.abs()\
                .mean((1,2,3), keepdim=True)\
                .expand_as(weights).clone()
        alpha[weights.lt(-1.0)] = 0
        alpha[weights.gt(1.0)] = 0

        alpha.mul_(grad_output)
        
        mean_grad = grad_output.mean((1,2,3), keepdim=True).expand(dim)

        return alpha.add(mean_grad) #type: ignore

class BinLinearParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights) -> torch.Tensor:
        ctx.save_for_backward(weights)
        saidas = weights[0].nelement()
        mean_weight = weights - weights.mean(1, keepdim=True).expand_as(weights)
        
        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .mean(1, keepdim=True)\
                .expand_as(weights)
        
        return clamped_weight.sign().mul(alpha)

    @staticmethod
    def backward(ctx, grad_output): #type: ignore
        weights, = ctx.saved_tensors
        saidas = weights[0].nelement() #num de saídas
        dim = weights.size() #dimensões

        alpha = weights.abs()\
                .mean(1, keepdim=True)\
                .expand(dim).clone()
        #derivada parcial de sign em função do i-esimo peso
        alpha[weights.lt(-1.0)] = 0
        alpha[weights.gt(1.0)] = 0

        # print(alpha)
        # print(weights)
        # print(grad_output)
        grad_input = grad_output.clone()
        #multiplica a derivada do custo em função do peso binarizada
        #pela derivada de sign em função do peso
        alpha.mul_(grad_input)

        mean_grad = grad_output.mean(1, keepdim=True).expand(dim)
        return alpha.add(mean_grad)  #type: ignore


class Conv2dBinary(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
    
    def forward(self, input):
        
        weight_binarized = BinConvParam.apply(self.weight)
        output = F.conv2d(
            input, weight_binarized, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return output
    
class LinearBinary(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        weight_binarized = BinLinearParam.apply(self.weight)

        output = F.linear(input, weight_binarized, self.bias)
        return output
    
class C3(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_maps = [
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 4, 5], [0, 1, 5],
            [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 3, 4, 5], [0, 1, 4, 5], [0, 1, 2, 5],
            [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
            [0, 1, 2, 3, 4, 5]
        ]

        self.conv_layer = nn.ModuleList()

        for input in self.feature_maps:
            conv = Conv2dBinary(in_channels=len(input),out_channels=1,kernel_size=5)
            self.conv_layer.append(conv)
        
    def forward(self, input):
        c3 = []
        for i, maps in enumerate(self.feature_maps):

            S2 = input[:, maps, :,:]
            resultado = self.conv_layer[i](S2)
            c3.append(resultado)

        return torch.cat(c3, dim=1)
