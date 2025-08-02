import torch
import torch.nn as nn
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=False),#mudar o canal de entrada pare 3 caso usar a cifar10
            nn.BatchNorm2d(6),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels=6,out_channels=16, kernel_size=5, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(),
        )

        self.linearLayer = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1)
        out = self.linearLayer(out)
        return out