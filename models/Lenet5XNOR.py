import torch
import torch.nn as nn
from .util_models import Binarize, Conv2dBinary, LinearBinary

def updateBinaryGradWeight(param):
        with torch.no_grad():
            saidas = param[0].nelement() #num de saídas
            dim = param.size() #dimensões
            # print(f"saídas:{saidas}")
            # print(f"dimensão:{dim}")
            if len(dim) == 4:
                alpha = param.abs()\
                        .mean(dim=(1,2,3), keepdim=True)\
                        .expand(dim).clone()
            elif len(dim) == 2:
                alpha = param.abs().mean(1, keepdim=True).expand(dim).clone()

            alpha[param.lt(-1.0)] = 0 #type: ignore
            alpha[param.gt(1.0)] = 0 #type: ignore

            alpha.mul_(param.grad) #type: ignore #alpha * gradiente dos pesos
            
            alpha_add = param.grad.div(saidas)

            param.grad = alpha.add(alpha_add).mul(1.0-1.0/dim[1]) #type: ignore 
            #.mul(1.0-1.0/s[1]) heuristica: input plane scaling

class LeNet5XNOR(nn.Module):
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
            nn.Hardtanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(50),
        )
        
        # self.layer3 = nn.Sequential(
        #     nn.BatchNorm2d(50),
        #     Binarize(),
        #     nn.Conv2d(in_channels=50, out_channels=120, kernel_size=5),
        #     nn.Hardtanh(),
        # )

        self.layer4 = nn.Sequential(
            Binarize(),
            LinearBinary(50*5*5, 500),
            nn.Hardtanh(),
        )

        self.fc = nn.Linear(500, num_classes)


        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)

            if isinstance(m, Conv2dBinary) or isinstance(m, LinearBinary):
                m.weight.register_post_accumulate_grad_hook(updateBinaryGradWeight)


    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)

        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.layer4(out)
        out = self.fc(out)
        return out
