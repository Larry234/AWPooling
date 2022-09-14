from re import T
import torch
import torch.nn as nn
from .awpooling import AWPool2d
from torch.autograd import Variable


class VGG11(nn.Module):
    def __init__(self, num_class=100, init_weights=True):
        super(VGG11, self).__init__()
        self.t = Variable(torch.Tensor([1., 1., 1., 1., 1.]), requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        self.aw1 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[0])

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        self.aw2 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[1])

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        self.aw3 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[2])

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.aw4 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[3])

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.aw5 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[4])

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
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), -1)
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

class VGG13(nn.Module):
    def __init__(self, num_class=100, init_weights=True):
        super(VGG13, self).__init__()
        self.t = Variable(torch.Tensor([1., 1., 1., 1., 1.]), requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        self.aw1 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[0])

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        self.aw2 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[1])

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        self.aw3 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[2])

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.aw4 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[3])

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.aw5 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[4])

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
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), -1)
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

class VGG16(nn.Module):
    def __init__(self, num_class=100, init_weights=True):
        super(VGG16, self).__init__()
        self.t = Variable(torch.Tensor([1., 1., 1., 1., 1.]), requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        self.aw1 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[0])

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        self.aw2 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[1])

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        self.aw3 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[2])

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.aw4 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[3])

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.aw5 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[4])

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
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), -1)
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

class VGG19(nn.Module):
    def __init__(self, num_class=100, init_weights=True):
        super(VGG19, self).__init__()
        self.t = Variable(torch.Tensor([1., 1., 1., 1., 1.]), requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        self.aw1 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[0])

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        self.aw2 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[1])

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        self.aw3 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[2])

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.aw4 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[3])

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.aw5 = AWPool2d(kernel_size=2, stride=2, temperature=self.t[4])

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
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), -1)
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

