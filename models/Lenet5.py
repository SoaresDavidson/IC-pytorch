import torch
import torch.nn as nn


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
            conv = nn.Conv2d(in_channels=len(input),out_channels=1,kernel_size=5)
            self.conv_layer.append(conv)
        
    def forward(self, input):
        c3 = []
        for i, maps in enumerate(self.feature_maps):

            S2 = input[:, maps, :,:]
            resultado = self.conv_layer[i](S2)
            c3.append(resultado)

        return torch.cat(c3, dim=1)

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.convLayer = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),#mudar o canal de entrada pare 3 caso usar a cifar10
            nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = 2, stride = 2)
        )

        self.convLayer1 = nn.Sequential(
            nn.Conv2d(6,16,5),
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