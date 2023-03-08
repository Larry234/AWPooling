import torch
import torch.nn as nn
from models.awpooling import *
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from models.vggaw import VGG11AW, VGG11AWT
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='tensorboard path', default='Gradient_analysis')
parser.add_argument('--scale', help='scale of delta', type=int, default=1)
parser.add_argument('--seed', help='random seed to keep initial weight equal',type=int, default=777)


class Net(nn.Module):
    def __init__(self, num_class=200):
        super(Net, self).__init__()
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
            # nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(),
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
        self.aw1.t = t

if __name__ == '__main__':

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    delta = 1
    for i in range(args.scale):
        delta *= 0.1

    writer = SummaryWriter(log_dir=os.path.join(args.logdir, f'delta={delta}'))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ds = ImageFolder(root='/home/larry/Datasets/tiny-imagenet-200/train', transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]))
    loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=128, num_workers=2)

    model = Net()
    params = [{'params': p, 'lr': 0.1} for n, p in model.named_parameters() if 'aw' not in n]
    optimizer = torch.optim.SGD(params)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i, data in enumerate(loader):
        images, label = data
        images = images.to(device)
        label = label.to(device)

        # caculate numercial difference
        with torch.no_grad():
            model.aw1.t.requires_grad = False
            origin_t = model.aw1.t.item()

            # forward
            model.aw1.t.copy_(torch.tensor(origin_t + delta))
            logits = model(images)
            forward_loss = criterion(logits, label)

            # backward
            model.aw1.t.copy_(torch.tensor(origin_t - delta))
            logits = model(images)
            backward_loss = criterion(logits, label)

        model.aw1.t.copy_(torch.tensor(origin_t))

        # Automatic differentiation
        model.aw1.t.requires_grad = True
        logits = model(images)
        base_loss = criterion(logits, label)
        base_loss.backward()

        # plot gradient on tensorboard
        writer.add_scalars('Gradient analysis', {
            'Automatic differentiation': model.aw1.t.grad,
            'Forward difference': (forward_loss - base_loss) / delta,
            'Backward difference': (base_loss - backward_loss) / delta,
            'Central difference': (forward_loss - backward_loss) / (2 * delta),
        }, i)

        # update model parameters
        optimizer.step()
        optimizer.zero_grad()
        model.aw1.t.grad.zero_()

    print(f'{delta} done!')