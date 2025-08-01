import torch
import torch.nn as nn
from .util_models import activation_quantize_fn, Conv2dQ, LinearQ, quantize_gradient
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96,kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),

            activation_quantize_fn(a_bit=2),
            Conv2dQ(in_channels=96, out_channels=256,kernel_size=5, stride=1, padding=2, k_bits=1),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            
            activation_quantize_fn(a_bit=2),
            Conv2dQ(in_channels=256, out_channels=384,kernel_size=3, stride=1, padding=1, k_bits=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            activation_quantize_fn(a_bit=2),
            Conv2dQ(in_channels=384, out_channels=384,kernel_size=3, stride=1, padding=1, k_bits=1),
            # nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            activation_quantize_fn(a_bit=2),
            Conv2dQ(in_channels=384, out_channels=256,kernel_size=3, stride=1, padding=1, k_bits=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(output_size=6)
        )
        self.classifier = nn.Sequential(
            activation_quantize_fn(a_bit=2),
            LinearQ(in_features=9216, out_features=4096, k_bits=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            activation_quantize_fn(a_bit=2),
            LinearQ(in_features=4096, out_features=4096, k_bits=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, Conv2dQ) or isinstance(m, LinearQ):
                # print(m)
                m.weight.register_hook(quantize_gradient)


    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1)
        out = self.classifier(out)
        return out
