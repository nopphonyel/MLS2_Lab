from torch import cuda, no_grad, hub
from torch.utils.data import random_split, DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os

# MUST HAVE!
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CIFAR10(root='./data', train=True, download=True, transform=preprocess)

trainset, valset = random_split(trainset, [40000, 10000])
testset = CIFAR10(root='./data', train=False, download=True, transform=preprocess)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=3)
valloader = DataLoader(valset, batch_size=4, shuffle=True, num_workers=3)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=3)
# To see the structure of network, uncomment this line

import time
import copy

device = "cuda:0"

import torch
from custom_model.handmade_se import handmade_se

model = handmade_se()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
data_loader_dict = {
    'train': trainloader,
    'val': valloader
}

from util import Trainer
print("Architect : 0")
model.set_arch(0)
trainer = Trainer(model=model, dataloaders=data_loader_dict, criterion=criterion, optimizer=optimizer, device=device)
best_model, val_acc_history = trainer.train_model()
