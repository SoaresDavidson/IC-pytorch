import torch
import torch.nn as nn
from .util_models import activation_quantize_fn, Conv2dQ, LinearQ, quantize_gradient

class LenetQ(nn.Module):
    def __init__(self, num_classes, w_bits = 1, a_bits = 2, g_bits = 6):
        super().__init__()
    
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0, bias=False),#mudar o canal de entrada pare 3 caso usar a cifar10
            nn.BatchNorm2d(20),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            activation_quantize_fn(a_bit=a_bits),
            Conv2dQ(in_channels=20,out_channels=50, kernel_size=5, k_bits=w_bits),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # activation_quantize_fn(a_bit=a_bits),
            # Conv2dQ(in_channels=16, out_channels=120, kernel_size=5, k_bits=w_bits),
            # nn.BatchNorm2d(120),
            # nn.ReLU(),
        )

        self.linear = nn.Sequential(
            activation_quantize_fn(a_bit=a_bits),
            LinearQ(50*5*5, 500, k_bits=w_bits),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )
        

        for m in self.modules():
            if isinstance(m, Conv2dQ) or isinstance(m, LinearQ):
                # print(m)
                m.weight.register_hook(quantize_gradient)

        
    
    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out


