# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
from .util_models import Binarize, Conv2dBinary, LinearBinary
class LeNet5Binary(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.Convlayer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.Convlayer2 = nn.Sequential(
            nn.BatchNorm2d(6,eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            Conv2dBinary(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.Convlayer3 = nn.Sequential(
            nn.BatchNorm2d(16, eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            Conv2dBinary(in_channels=16, out_channels=120, kernel_size=5),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(120),
            LinearBinary(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )


    def forward(self, x):
        out = self.Convlayer1(x)
        out = self.Convlayer2(out)
        out = self.Convlayer3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        return out