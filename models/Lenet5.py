import torch
import torch.nn as nn
from .util_models import C3
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.convLayer = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),#mudar o canal de entrada pare 3 caso usar a cifar10
            nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = 2, stride = 2)
        )

        self.convLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = 2, stride = 2)
        )

        self.convLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.Tanh(),
        )

        self.linearLayer = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        out = self.convLayer(x)
        out = self.convLayer1(out)
        out = self.convLayer2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linearLayer(out)
        return out