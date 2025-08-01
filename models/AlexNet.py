import torch
import torch.nn as nn

def cu(module,input,output):
    print(f"->{output.shape}")
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96,kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),

            nn.Conv2d(in_channels=96, out_channels=256,kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),

            nn.Conv2d(in_channels=256, out_channels=384,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

        # for m in self.modules():
        #     if isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Conv2d):
        #         m.register_forward_hook(cu)
        

    def forward(self, x):
        # print(x.shape)
        out = self.features(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out) 
        out = self.fc3(out)
        return out

