import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.awpooling import AWPool2d
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

import re
import os
import re
import argparse
from glob import glob
import pandas as pd
import json

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
TUNE_T_PATH = 'HPO/tiny-imagenet'

def load_pretrain(model, pretrain):
    
    model_dict = model.state_dict()
    delete_keys = [k for k in model_dict.keys() if 'batch' in k]
    # delete batch_tracked
    for k in delete_keys:
        del model_dict[k]

    param_value = list(pretrain.values())
    index = 0
    
    for k in model_dict.keys():
        if 'aw' in k or 'classifier' in k:
            continue
        model_dict[k] = param_value[index]
        index += 1

    model.load_state_dict(model_dict, strict=False)

def get_loader(root):
    
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


def train_model(config, data=None): 
    assert torch.cuda.is_available()
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    train_loader, val_loader = get_loader(args.data)
    
    save_root = '/root/notebooks/nfs/work/larry.lai/AWPooling/checkpoints/rs' + config['arch']
    os.makedirs(save_root, exist_ok=True)
    model = get_network(net=config['arch'], num_class=config['num_class'])
    model.set_temperature(config)
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    best_acc = 0
    running_loss = 0
    
    for epoch in range(config['epochs']):
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
        session.report({"epoch": epoch, "accuracy": float(acc), "loss": running_loss / len(val_loader)}, checkpoint=checkpoint)
    
    
def train_from_pretrain(config, data=None):
    assert torch.cuda.is_available()
    
    save_root = '/home/larry/AWPooling/baysopt'
    os.makedirs(save_root, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_loader(args.data)
    
    save_root = os.path.join(TUNE_T_PATH, args.arch)
    os.makedirs(save_root, exist_ok=True)
    
    # load model and pretrain weight
    model = get_network(config['arch'], config['num_class'])
    load_pretrain(model, data)
    model.set_temperature(config)
    model.to(device) 
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    best_acc = 0
    running_loss = 0

    
    for epoch in range(config['epochs']):
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
        
        torch.save(model.state_dict(), os.path.join(save_root, 'last.pt'))
        checkpoint = Checkpoint.from_directory(save_root)
        acc = corrects / len(val_loader.dataset)
        
        acc = acc.data.cpu().numpy()
        session.report({"epoch": epoch, "accuracy": float(acc), "loss": running_loss / len(val_loader)}, checkpoint=checkpoint)
        
def compute_result(path):
    exp_path = glob(os.path.join(path, '*'))
    tems = []
    accs = []

    for p in exp_path:
        if not os.path.isdir(p) or 'error.pkl' in os.listdir(p):
            continue
        trial_name = ""
        csv_f = pd.read_csv(os.path.join(p, 'progress.csv'))
        accs.append(max(csv_f['accuracy']))
        f = open(os.path.join(p, 'params.json'))
        json_f = json.load(f)

        params = list(json_f.keys())[-5:]
        values = list(json_f.values())[-5:]
        for t, v in zip(params, values):
            trial_name += f'{t}={v},'
        tems.append(trial_name)

    df = pd.DataFrame({'tems': tems, 'accs': accs})
    df = df.sort_values(by=['accs'], ascending=False)
    df.to_csv(os.path.join(path, 'best.csv'), index=False)
    
    
        
    
def main(args):
    
    search_space = {
        "t0": tune.uniform(1e-5, 20),
        "t1": tune.uniform(1e-5, 20),
        "t2": tune.uniform(1e-5, 20),
        "t3": tune.uniform(1e-5, 20),
        "t4": tune.uniform(1e-5, 20),
        "arch": args.arch,
        "num_class": 200,
        "epochs": args.epochs,
        "data": args.data,
    }
    
    # allocate trial resources
    trainable = tune.with_resources(train_model, {'gpu': args.gpus, 'cpu': args.cpus})
    
    # define trail scheduler
    asha_scheduler = ASHAScheduler(
        time_attr='epoch',
        metric='accuracy',
        mode='max',
        grace_period=70,
    )
    
    tune_config = tune.TuneConfig(
        num_samples=args.num_samples,
        scheduler=asha_scheduler,
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute='accuracy',
        checkpoint_score_order='max'
    )
    
    run_config = RunConfig(
        name=args.arch,
        local_dir=args.exp,
        checkpoint_config=checkpoint_config,
    )

    tuner = tune.Tuner(
        trainable,
        tune_config=tune_config,
        run_config=run_config,
        param_space=search_space,
    )
    
    results = tuner.fit()
    
    compute_result(os.path.join(args.exp, args.arch))
    

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', help='model architecture', type=str, default='vgg16aw')
    parser.add_argument('--data', help='path to dataset', type=str, default='/home/larry/Datasets/tiny-imagenet-200')
    parser.add_argument('--epochs', help='total epochs in each trial', type=int, default=60)
    parser.add_argument('--num-samples', help='iterations of bayesian optimization', type=int, default=30)
    parser.add_argument('--exp', help='path to save experiment result', type=str, default='HPO/tiny-imagenet/rs')
    parser.add_argument('--gpus', help='how many gpus can a trial use, fraction is excepted', type=float, default=1.)
    parser.add_argument('--cpus', help='how many cpus can a trial use, fraction is excepted', type=float, default=2.)
    
    args = parser.parse_args()
    main(args)