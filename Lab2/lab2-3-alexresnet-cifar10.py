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

trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
valloader = DataLoader(valset, batch_size=1, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# Import resnet model
import torch

resnet18 = models.resnet18()
resnet18.fc = torch.nn.Linear(512, 10)
resnet18.eval()

# To see the structure of network, uncomment this line
# print(resnet18)

import time
import copy

device = "cuda:1"


from custom_model.alexresnet import alexresnet

model = alexresnet()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
data_loader_dict = {
    'train': trainloader,
    'val': valloader
}

from util import Trainer

trainer = Trainer(model=model, dataloaders=data_loader_dict, criterion=criterion, optimizer=optimizer)
best_model, val_acc_history = trainer.train_model()
