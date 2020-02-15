import torch
from torchvision import transforms
from cervix_datasets import CervixDataset
from PIL import Image
import urllib
import os

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

preprocess = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

from util import Trainer
from torchvision import models
from torch.utils.data import random_split, DataLoader

device = "cuda:1"

dataset = CervixDataset(root='./', csv_path="table_label_v2.csv", transform=preprocess)
# print(len(dataset))
trainset, valset = random_split(dataset, [75, 23])
trainloader = DataLoader(trainset, shuffle=True)
valloader = DataLoader(valset, shuffle=True)

model = models.resnet50(pretrained=False)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
data_loader_dict = {
    'train': trainloader,
    'val': valloader
}
model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
model = model.to(device)

trainer = Trainer(model=model, dataloaders=data_loader_dict, criterion=criterion, optimizer=optimizer, device=device,
                  filename="RESNET50_ADAM_Cervix.pth", num_epochs=100)
best_model, val_acc_history = trainer.train_model()
