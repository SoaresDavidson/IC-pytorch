import torch
import torch.nn as nn

def conv_block(output):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(inplace=True),
        nn.LazyConv2d(output, kernel_size=3, padding=1)
    )

def transition_block(output):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(inplace=True),
        nn.LazyConv2d(output, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(conv_block(num_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for F in self.net:
            out = F(x)
            x = torch.cat((out, x), dim=1)
        return x

class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate= 32, arch = (4, 4, 4, 4)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        for i, num_convs in enumerate(arch):
            self.net.add_module(f"dense_blk{i+1}", DenseBlock(num_convs, growth_rate))
            
            num_channels = num_convs * growth_rate
            if i != len(arch) - 1:
                num_channels //=2
                self.net.add_module(f"transition_layer{i+1}", transition_block(num_channels))

        self.final = nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        out = self.net(x)
        return self.final(out)
        


