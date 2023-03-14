import torch
import torch.nn as nn
from models.awpooling import *
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from models.vggaw import VGG11AW, VGG11AWT
import os
import argparse

from utils import get_network

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Dataset path', default='/home/larry/Datasets/tiny-imagenet-200')
parser.add_argument('--arch', help='model architecture', type=str)
parser.add_argument('--epochs', help='number of epochs', type=int, default=60)
parser.add_argument('--batch-size', help='number of images in one iteration', type=int, default=128)
parser.add_argument('--scale', help='scale of delta', type=int, default=1)
parser.add_argument('--seed', help='random seed to keep initial weight equal',type=int, default=777)
parser.add_argument('--logdir', help='tensorboard path', default='Gradient_analysis')


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
        self.aw1.t = t

if __name__ == '__main__':

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    delta = 1
    for i in range(args.scale):
        delta *= 0.1
    
    arch = args.arch if args.arch != None else 'SimpleNet'
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, arch, f'delta={delta}'))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_ds = ImageFolder(
        root=traindir, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )

    val_ds = ImageFolder(
        root=valdir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=128, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, num_workers=2, pin_memory=True)

    if args.arch:
        model = get_network(args.arch, num_class=200)
    else:
        model = Net()

    params = [{'params': p, 'lr': 0.1} for n, p in model.named_parameters() if 'aw' not in n]
    optimizer = torch.optim.SGD(params)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # training process
    for i, data in enumerate(train_loader):
        images, label = data
        images = images.to(device)
        label = label.to(device)

        logits = model(images)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation process
    for i, data in enumerate(val_loader):
        # freeze batchNorm layer and Dropout layer
        model.eval()

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

        exact_diff = model.aw1.t.grad
        f_diff = (forward_loss - base_loss) / delta
        b_diff = (base_loss - backward_loss) / delta
        c_diff = (forward_loss - backward_loss) / (2 * delta)

        print(f'iter {i}: ' \
              f'exact grad: {exact_diff.item(): .8f}\n' \
              f'forward diff: {f_diff.item(): .8f}\n'  \
              f'backward diff: {b_diff.item(): .8f}\n'  \
              f'central diff: {c_diff.item()}\n')

        # plot gradient on tensorboard
        writer.add_scalars('Gradient analysis', {
            'Automatic differentiation': exact_diff,
            'Forward difference': f_diff,
            'Backward difference': b_diff,
            'Central difference': c_diff,
        }, i)

        f_diff = torch.abs(exact_diff - f_diff) if torch.sign(exact_diff) == torch.sign(f_diff) else -torch.abs(exact_diff - f_diff)
        b_diff = torch.abs(exact_diff - b_diff) if torch.sign(exact_diff) == torch.sign(b_diff) else -torch.abs(exact_diff - b_diff)
        f_diff = torch.abs(exact_diff - c_diff) if torch.sign(exact_diff) == torch.sign(c_diff) else -torch.abs(exact_diff - c_diff)

        writer.add_scalars('Difference', {
            'Forward': f_diff,
            'Backward': b_diff,
            'Central': c_diff
        }, i)

        # update model parameters
        optimizer.step()
        optimizer.zero_grad()
        model.aw1.t.grad.zero_()

    print(f'{delta} done!')