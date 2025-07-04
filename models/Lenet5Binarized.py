# Load in relevant libraries, and alias where appropriate
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
    
class BinConvParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        saidas = x[0].nelement()
        mean_weight = x - x.mean(1, keepdim=True).expand_as(x)

        
        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .sum(3, keepdim=True)\
                .sum(2, keepdim=True)\
                .sum(1, keepdim=True)\
                .div(saidas).expand_as(x)
        
        return clamped_weight.sign().mul(alpha)

    @staticmethod
    def backward(ctx, grad_output): #type: ignore
        param, = ctx.saved_tensors
        saidas = param[0].nelement() #num de saídas
        dim = param.size() #dimensões

        alpha = param.abs()\
                .sum(3, keepdim=True)\
                .sum(2, keepdim=True)\
                .sum(1, keepdim=True)\
                .div(saidas).expand_as(param).clone()
    
        alpha[param.lt(-1.0)] = 0
        alpha[param.gt(1.0)] = 0

        alpha.mul_(grad_output)
        alpha_add = param.sign().mul(grad_output)
    
        alpha_add = alpha_add.sum(3, keepdim=True)\
                                .sum(2, keepdim=True)\
                                .sum(1, keepdim=True)\
                                .div(saidas).expand_as(param)
        
        alpha_add = alpha_add.mul(param.sign())
        return alpha.add(alpha_add).mul(1.0-1.0/dim[1]).mul(saidas) #type: ignore

class BinLinearParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        mean_weight = x - x.mean(1, keepdim=True).expand_as(x)
        
        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .sum(1, keepdim=True)\
                .div(2).expand_as(x)
        
        return clamped_weight.sign().mul(alpha)

    @staticmethod
    def backward(ctx, grad_output): #type: ignore
        param, = ctx.saved_tensors
        saidas = param[0].nelement() #num de saídas
        dim = param.size() #dimensões

        alpha = param.abs()\
                .sum(1, keepdim=True)\
                .div(saidas).expand(dim).clone()
    
        alpha[param.lt(-1.0)] = 0
        alpha[param.gt(1.0)] = 0

        grad_input = grad_output.clone()
        alpha.mul_(grad_input)
        alpha_add = param.sign().mul(grad_input)
    
        alpha_add = alpha_add.sum(1, keepdim=True)\
                            .div(saidas).expand(dim)
        
        alpha_add = alpha_add.mul(param.sign())
        return alpha.add(alpha_add).mul(1.0-1.0/dim[1]) #type: ignore

class Binarize(nn.Module):
    def __init__(self):
        super(Binarize, self).__init__()

    def forward(self, x):
        return BinFunction.apply(x)

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

class LeNet5Binary(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            Conv2dBinary(in_channels=20, out_channels=50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        
        self.layer3 = nn.Sequential(
            nn.BatchNorm1d(50*5*5, eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            LinearBinary(1250, 500),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.BatchNorm1d(500, eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            nn.Linear(500, num_classes),
            # nn.ReLU()
        )

        # self.fc2 = nn.Linear(84, num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)


    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)

        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.fc2(out)
        return out

# from torch.autograd import gradcheck

# x = torch.randn((1, 3, 5, 5), dtype=torch.double, requires_grad=True)
# test = BinConvParam.apply

# gradcheck(test, (x,), eps=1e-6, atol=1e-4)  # Should return True
