import torch
import torch.nn as nn
from .util_models import Binarize, Conv2dBinary, LinearBinary

class LeNet5XNOR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(6, eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            Conv2dBinary(in_channels=6, out_channels=16, kernel_size=5),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(16, eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            Conv2dBinary(16, 120, 5),
            # nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.BatchNorm1d(120, eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            LinearBinary(120, 84),
            # nn.ReLU()
        )

        self.fc = nn.Linear(84, num_classes)

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
        out = self.layer3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.layer4(out)
        out = self.fc(out)
        return out
