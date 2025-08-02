import torch
import torch.nn as nn
from .util_models import Binarize, Conv2dBinary, LinearBinary


class LeNet5XNOR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
 
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            Conv2dBinary(in_channels=20, out_channels=50, kernel_size=5),
            nn.Hardtanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.dense = nn.Sequential(
            nn.BatchNorm1d(50*5*5),
            Binarize(),
            LinearBinary(50*5*5, 500),
            nn.Hardtanh(),
        )

        self.fc = nn.Linear(500, num_classes)


        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)

            # if isinstance(m, Conv2dBinary) or isinstance(m, LinearBinary):
            #     m.weight.register_post_accumulate_grad_hook(updateBinaryGradWeight)

        

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)

        out = self.features(x)
        # out = self.layer3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        out = self.fc(out)
        return out
