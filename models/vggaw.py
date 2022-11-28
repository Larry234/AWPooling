import torch
import torch.nn as nn
from torch.autograd import Variable
from .awpooling import *


class VGG11(nn.Module):
    def __init__(self, num_class=100, init_weights=True):
        super(VGG11, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.aw1 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.aw2 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.aw3 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.aw4 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.aw5 = AWPool2d_(kernel_size=2, stride=2)
        self.globalavg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.aw1(x)
        x = self.conv2(x)
        x = self.aw2(x)
        x = self.conv3(x)
        x = self.aw3(x)
        x = self.conv4(x)
        x = self.aw4(x)
        x = self.conv5(x)
        x = self.aw5(x)
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
                
    def set_temperature(self, t1, t2, t3, t4, t5):
        self.aw1.t = nn.Parameter(torch.FloatTensor([t1]))
        self.aw2.t = nn.Parameter(torch.FloatTensor([t2]))
        self.aw3.t = nn.Parameter(torch.FloatTensor([t3]))
        self.aw4.t = nn.Parameter(torch.FloatTensor([t4]))
        self.aw5.t = nn.Parameter(torch.FloatTensor([t5]))
    
    def disable_t(self):
        self.aw1.t.requires_grad = False
        self.aw2.t.requires_grad = False
        self.aw3.t.requires_grad = False
        self.aw4.t.requires_grad = False
        self.aw5.t.requires_grad = False
        

    def enable_t(self, index):
        if index == 1:
            self.aw1.t.requires_grad = True
        elif index == 2:
            self.aw2.t.requires_grad = True
        elif index == 3:
            self.aw3.t.requires_grad = True
        elif index == 4:
            self.aw4.t.requires_grad = True
        elif index == 5:
            self.aw5.t.requires_grad = True
        

# ====================================== VGG13 ===================================================
class VGG13(nn.Module):
    def __init__(self, num_class=100, init_weights=True):
        super(VGG13, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.aw1 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.aw2 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.aw3 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.aw4 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.aw5 = AWPool2d_(kernel_size=2, stride=2)
        self.globalavg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.aw1(x)
        x = self.conv2(x)
        x = self.aw2(x)
        x = self.conv3(x)
        x = self.aw3(x)
        x = self.conv4(x)
        x = self.aw4(x)
        x = self.conv5(x)
        x = self.aw5(x)
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

                
    def disable_t(self):
        self.aw1.requires_grad = False
        self.aw2.requires_grad = False
        self.aw3.requires_grad = False
        self.aw4.requires_grad = False
        self.aw5.requires_grad = False
        
        
    def enable_t(self, index):
        if index == 1:
            self.aw1.requires_grad = True
        elif index == 2:
            self.aw2.requires_grad = True
        elif index == 3:
            self.aw3.requires_grad = True
        elif index == 4:
            self.aw4.requires_grad = True
        elif index == 5:
            self.aw5.requires_grad = True
                
# ====================================== VGG16 ===================================================
class VGG16(nn.Module):
    def __init__(self, num_class=100, init_weights=True):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.aw1 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.aw2 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        self.aw3 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.aw4 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.aw5 = AWPool2d_(kernel_size=2, stride=2)
        self.globalavg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.aw1(x)
        x = self.conv2(x)
        x = self.aw2(x)
        x = self.conv3(x)
        x = self.aw3(x)
        x = self.conv4(x)
        x = self.aw4(x)
        x = self.conv5(x)
        x = self.aw5(x)
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

                
                
    def disable_t(self):
        self.aw1.requires_grad = False
        self.aw2.requires_grad = False
        self.aw3.requires_grad = False
        self.aw4.requires_grad = False
        self.aw5.requires_grad = False
        
        
    def enable_t(self, index):
        if index == 1:
            self.aw1.requires_grad = True
        elif index == 2:
            self.aw2.requires_grad = True
        elif index == 3:
            self.aw3.requires_grad = True
        elif index == 4:
            self.aw4.requires_grad = True
        elif index == 5:
            self.aw5.requires_grad = True


# ====================================== VGG19 ===================================================
class VGG19(nn.Module):
    def __init__(self, num_class=100, init_weights=True):
        super(VGG19, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.aw1 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.aw2 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        self.aw3 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.aw4 = AWPool2d_(kernel_size=2, stride=2)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.aw5 = AWPool2d_(kernel_size=2, stride=2)
        self.globalavg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.aw1(x)
        x = self.conv2(x)
        x = self.aw2(x)
        x = self.conv3(x)
        x = self.aw3(x)
        x = self.conv4(x)
        x = self.aw4(x)
        x = self.conv5(x)
        x = self.aw5(x)
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
    
    def disable_t(self):
        self.aw1.requires_grad = False
        self.aw2.requires_grad = False
        self.aw3.requires_grad = False
        self.aw4.requires_grad = False
        self.aw5.requires_grad = False
    
    def enable_t(self, index):
        if index == 1:
            self.aw1.requires_grad = True
        elif index == 2:
            self.aw2.requires_grad = True
        elif index == 3:
            self.aw3.requires_grad = True
        elif index == 4:
            self.aw4.requires_grad = True
        elif index == 5:
            self.aw5.requires_grad = True
            
            
class VGG11AW(nn.Module):
    def __init__(self, num_class=100, init_weights=True):
        super(VGG11AW, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.globalavg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        
        self.aws = {f't{i}': AWPool2d(kernel_size=2, stride=2, temperature=1) for i in range(5)}
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.aws['t0'](x)
        x = self.conv2(x)
        x = self.aws['t1'](x)
        x = self.conv3(x)
        x = self.aws['t2'](x)
        x = self.conv4(x)
        x = self.aws['t3'](x)
        x = self.conv5(x)
        x = self.aws['t4'](x)
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
                
    def set_temperature(self, config: dict):
        for k, v in config.items():
            if k in self.aws.keys():
                self.aws[k].t = v
    