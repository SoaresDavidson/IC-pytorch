import torch
import torch.nn as nn
from .util_models import Binarize, Conv2dBinary, LinearBinary

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96,kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.BatchNorm2d(96),
            Binarize(),
            Conv2dBinary(in_channels=96, out_channels=256,kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.BatchNorm2d(256),
            Binarize(),
            Conv2dBinary(in_channels=256, out_channels=384,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm2d(384),
            Binarize(),
            Conv2dBinary(in_channels=384, out_channels=384,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(384),
            Binarize(),
            Conv2dBinary(in_channels=384, out_channels=256,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(256),
            Binarize(),
            LinearBinary(in_features=256, out_features=4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),

            nn.BatchNorm1d(4096),
            Binarize(),
            LinearBinary(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
        )

        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)


        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)



    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out) 
        out = self.fc3(out)
        return out

