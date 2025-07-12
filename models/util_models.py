import torch
import torch.nn as nn
import torch.nn.functional as F

class BinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class Binarize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return BinFunction.apply(input)


class BinConvInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        temp = input.clone()
        _, features,width,height = input.size()

        alpha = temp.abs().mean(1, keepdim=True).expand_as(input).clone() 

        k = torch.ones(width , height, device=input.device).div(width/height)
        k = k.expand(features,1,width,height)

        beta = F.conv2d(alpha, k, groups=features)
        return beta * Binarize()(input)

    @staticmethod
    def backward(ctx, grad_output): #type: ignore
        grad_bin = grad_output.sign()
        scaling = grad_output.amax((1,2,3), keepdim=True)

        return grad_bin * scaling

class BinarizeAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return BinConvInput.apply(input)

class BinWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output): # type: ignore #1 back 2 back
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input
    
class BinarizeWeights(nn.Module):
    def __init__(self):
        super(BinarizeWeights, self).__init__()

    def forward(self, input):
        return BinWeights.apply(input)
    
class BinConvParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        ctx.save_for_backward(weights)
        mean_weight = weights - weights.mean(1, keepdim=True).expand_as(weights)

        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .mean((1,2,3), keepdim=True)\
                .expand_as(weights)

        return BinarizeWeights()((clamped_weight).mul(alpha))

    @staticmethod
    def backward(ctx, grad_output): #type: ignore 3 back 4 back
        weights, = ctx.saved_tensors
        num_weights = weights.numel()
        # alpha = weights.abs()\
        #         .mean((1,2,3), keepdim=True)\
        #         .expand_as(weights).clone()
        
        # alpha[weights.lt(-1.0)] = 0
        # alpha[weights.gt(1.0)] = 0

        # alpha_grad = alpha.mul(grad_output)

        mean_grad = grad_output.div(num_weights)

        return mean_grad.add(grad_output) #+ alpha_grad #type: ignore
class Conv2dBinary(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
    
    def forward(self, input):
        weight_binarized = BinConvParam.apply(self.weight)

        output = F.conv2d(
            input, weight_binarized, self.bias, self.stride, self.padding, self.dilation, self.groups #type: ignore
        ) 
        return output

class BinLinearParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights) -> torch.Tensor:
        ctx.save_for_backward(weights)
        mean_weight = weights - weights.mean(1, keepdim=True).expand_as(weights)
        
        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .mean(1, keepdim=True)\
                .expand_as(weights)
        
        return BinarizeWeights()((clamped_weight).mul(alpha))
    
    @staticmethod
    def backward(ctx, grad_output): #type: ignore 5 back 6 back
        weights, = ctx.saved_tensors
        num_weights = weights.numel()

        # alpha = weights.abs().mean(1,keepdim=True).expand_as(weights).clone()

        # alpha[weights.gt(1.0)] = 0
        # alpha[weights.lt(-1.0)] = 0

        # alpha_grad = alpha.mul(grad_output)

        mean_grad = grad_output.div(num_weights)

        return mean_grad.add(grad_output) #+ alpha_grad #type: ignore
class LinearBinary(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        weight_binarized = BinLinearParam.apply(self.weight)

        output = F.linear(input, weight_binarized, self.bias) #type: ignore
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
