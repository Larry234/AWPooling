import torch
import torch.nn as nn
from .awpooling import AWPool2d_

class SimpleNet(nn.Module):
    def __init__(self, num_class=200):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.aw1 = AWPool2d_()

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        # )
        # self.aw2 = AWPool2d_()
    
        self.globalavg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_class),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.aw1(x)
        x = self.globalavg(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def set_t(self, t):
        # modify parameter value in-place with no gradient
        with torch.no_grad():
            self.aw1.t.copy_(torch.tensor(t))
    
    def get_t(self):
        return self.aw1.t.item()