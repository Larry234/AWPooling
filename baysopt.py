import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.awpooling import AWPool2d
from models.vggaw import VGG11AW, VGG16AW
from utils import get_network

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.air.config import RunConfig, CheckpointConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch

import os
import argparse


class Trainable(tune.Trainable):
    def setup(self, config: dict):
        self.t = config['temperature']
        self.obj = config['obj']

    def step(self):
        pass

    def get_loader(self):
        pass

    def save_checkpoint(self, tmp_checkpoint_dir):
        pass

    def load_checkpoint(self, tmp_checkpoint_dir):
        pass



def get_loader(root='/root/notebooks/nfs/work/dataset/tiny-imagenet-200'):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    
    train_ds = datasets.ImageFolder(
        traindir,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    val_ds = datasets.ImageFolder(
        valdir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    print("Loading data...")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def train_model(config): 
    
    assert torch.cuda.is_available()
    
    save_root = '/home/larry/AWPooling/baysopt'
    os.makedirs(save_root, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    train_loader, val_loader = get_loader('/home/larry/Datasets/tiny-imagenet-200')
    
    model = get_network(net=config['arch'], num_class=200)
    model.set_temperature(config)
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    best_acc = 0
    running_loss = 0
    
    for epoch in range(90):
        model.train()
        for data in train_loader:
            image, label = data
            image = image.to(device)
            label = label.to(device)
            
            logits = model(image)
            loss = criterion(logits, label)
            running_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        print(f"Epoch[{epoch + 1}/90]: loss {running_loss/len(train_loader):.4f}")
        
        running_loss = 0
        corrects = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                image, label = data
                image = image.to(device)
                label = label.to(device)
                
                logits = model(image)
                loss = criterion(logits, label)
                running_loss += loss.item()
                
                _, pred = logits.max(dim=1)
                corrects += pred.eq(label).sum()
        
        acc = corrects / len(val_loader.dataset)
        
        best_acc = acc if acc > best_acc else best_acc
        
        acc = acc.data.cpu().numpy()
        checkpoint = Checkpoint.from_directory(save_root)
        session.report({"accuracy": float(acc), "loss": running_loss / len(val_loader)}, checkpoint=checkpoint)
    
    print(f"trial {session.get_trial_name()}: best acc: {best_acc:.4f}")
    print("Finish training")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', help='model architecture', type=str, default='vgg16aw')
    parser.add_argument('--goal', help='objective function to optimize', type=str, default='accuracy', choices=['loss', 'accuracy'])
    parser.add_argument('--kind', help='acquisition function kind', type=str, default='ucb', choices=['ucb', 'ei', 'pi'])
    parser.add_argument('--kappa', help='balance of exploration and exploitation in upper confidence bound', type=float, default=2.5)
    parser.add_argument('--xi', help='balance of exploration and exploitation in expected improvement and probability of improvement', type=float, default=0.0)
    parser.add_argument('--iter', help='iterations of bayesian optimization', type=int, default=30)
    parser.add_argument('--path', help='experiment path', type=str, default='exp')

    args = parser.parse_args()

    search_space = {
        "arch": args.arch,
        "t0": tune.uniform(1e-5, 0.5),
        "t1": tune.uniform(1e-5, 1),
        "t2": tune.uniform(1, 5),
        "t3": tune.uniform(2, 10),
        "t4": tune.uniform(2, 10),
    }

    algo = BayesOptSearch(metric='loss', mode='min', utility_kwargs={'kind': args.kind, 'kappa': args.kappa, 'xi': args.xi})
    tune_config = tune.TuneConfig(
        num_samples=args.iter,
        search_alg=algo
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute='accuracy',
        checkpoint_score_order='max'
    )

    tuner = tune.Tuner(
        tune.with_resources(train_model, {'gpu': 1, 'cpu': 4}),
        tune_config=tune_config,
        run_config=RunConfig(local_dir='./test_run', name=args.path, checkpoint_config=checkpoint_config),
        param_space=search_space,
    )
    results = tuner.fit()