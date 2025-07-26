import torch
import torch.nn as nn
from .util_models import activation_quantize_fn, Conv2dQ, LinearQ, QuantizeGradient

class LenetQ(nn.Module):
    def __init__(self, num_classes, w_bits = 1, a_bits = 2, g_bits = 6):
        super().__init__()
    
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),#mudar o canal de entrada pare 3 caso usar a cifar10
            nn.BatchNorm2d(6),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            Conv2dQ(in_channels=6,out_channels=16, kernel_size=5, k_bits=w_bits),
            QuantizeGradient(k_bits = g_bits),
            nn.BatchNorm2d(16),
            activation_quantize_fn(a_bit=a_bits),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            Conv2dQ(in_channels=16, out_channels=120, kernel_size=5, k_bits=w_bits),
            QuantizeGradient(k_bits = g_bits),
            nn.BatchNorm2d(120),
            activation_quantize_fn(a_bit=a_bits),
        )

        self.linearLayer = nn.Sequential(
            LinearQ(120, 84, k_bits=w_bits),
            QuantizeGradient(k_bits = g_bits),
            activation_quantize_fn(a_bit=a_bits),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1)
        out = self.linearLayer(out)
        return out


