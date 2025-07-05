
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        saidas = weights[0].nelement()
        mean_weight = weights - weights.mean(1, keepdim=True).expand_as(weights)

        
        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .sum(3, keepdim=True)\
                .sum(2, keepdim=True)\
                .sum(1, keepdim=True)\
                .div(saidas).expand_as(weights)
        
        return clamped_weight.sign().mul(alpha)

    @staticmethod
    def backward(ctx, grad_output): #type: ignore
        weights, = ctx.saved_tensors
        saidas = weights[0].nelement() #num de saídas
        dim = weights.size() #dimensões

        alpha = weights.abs()\
                .sum(3, keepdim=True)\
                .sum(2, keepdim=True)\
                .sum(1, keepdim=True)\
                .div(saidas).expand_as(weights).clone()
    
        alpha[weights.lt(-1.0)] = 0
        alpha[weights.gt(1.0)] = 0

        alpha.mul_(grad_output)
        alpha_add = weights.sign().mul(grad_output)
    
        alpha_add = alpha_add.sum(3, keepdim=True)\
                                .sum(2, keepdim=True)\
                                .sum(1, keepdim=True)\
                                .div(saidas).expand_as(weights)
        
        alpha_add = alpha_add.mul(weights.sign())
        return alpha.add(alpha_add).mul(1.0-1.0/dim[1]).mul(saidas) #type: ignore

class BinLinearParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        ctx.save_for_backward(weights)
        mean_weight = weights - weights.mean(1, keepdim=True).expand_as(weights)
        
        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .sum(1, keepdim=True)\
                .div(2).expand_as(weights)
        
        return clamped_weight.sign().mul(alpha)

    @staticmethod
    def backward(ctx, grad_output): #type: ignore
        weights, = ctx.saved_tensors
        saidas = weights[0].nelement() #num de saídas
        dim = weights.size() #dimensões

        alpha = weights.abs()\
                .sum(1, keepdim=True)\
                .div(saidas).expand(dim).clone()
    
        alpha[weights.lt(-1.0)] = 0
        alpha[weights.gt(1.0)] = 0
        # print(alpha)
        # print(weights)
        # print(grad_output)
        grad_input = grad_output.clone()
        alpha.mul_(grad_input)

        alpha_add = weights.sign().mul(grad_input)
        alpha_add = alpha_add.sum(1, keepdim=True)\
                            .div(saidas).expand(dim)
        
        alpha_add = alpha_add.mul(weights.sign())
        return alpha.add(alpha_add).mul(1.0-1.0/dim[1]).mul(saidas) #type: ignore


class Conv2dBinary(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
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

