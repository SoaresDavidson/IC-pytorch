import torch
import torch.nn as nn

class NiN(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        def nin_block(out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.LazyConv2d(out_channels, kernel_size, stride, padding), nn.ReLU(),
                nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
                nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU()
            )
            
        self.features = nn.Sequential(
            nin_block(out_channels=96,kernel_size=5, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.BatchNorm2d(96),

            nin_block(out_channels=256,kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.BatchNorm2d(256),

            nin_block(out_channels=384,kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),

            nin_block(num_classes ,kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten()
        )

    def forward(self, x):
        # print(x.shape)
        out = self.features(x)
        return out
    
