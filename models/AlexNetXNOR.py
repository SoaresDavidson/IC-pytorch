import torch
import torch.nn as nn
from .util_models import BinarizeAct

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.BatchNorm2d(64),
            BinarizeAct(),
            nn.Conv2d(in_channels=64, out_channels=192,kernel_size=5, stride=1, padding=2),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.BatchNorm2d(192, affine=True),
            BinarizeAct(),
            nn.Conv2d(in_channels=192, out_channels=384,kernel_size=3, stride=1, padding=1),
            nn.Hardtanh(inplace=True),
        )
        self.conv_layer4 = nn.Sequential(
            nn.BatchNorm2d(384, affine=True),
            BinarizeAct(),
            nn.Conv2d(in_channels=384, out_channels=256,kernel_size=3, stride=1, padding=1),
            nn.Hardtanh(inplace=True),
        )
        self.conv_layer5 = nn.Sequential(
            nn.BatchNorm2d(256, affine=True),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3, stride=1, padding=1),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # self.avgPoll = nn.AdaptiveAvgPool2d(output_size=6)
        self.fc1 = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(256, affine=True),
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        # out = self.avgPoll(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc3(out)
        return out

