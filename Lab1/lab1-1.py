import torch
# Import training data and stuff
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import urllib
import os

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

model = torch.hub.load('pytorch/vision:v0.5.0', 'alexnet', pretrained=False)
model.eval()

# Load CIFAR data sets (also do some transformation.... but why?)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = CIFAR10(root='./datasets', train=True, download=True, transform=transform)
tr_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

print(len(trainset))

testset = CIFAR10(root='./datasets', train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

CIFAR_Classes = (
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
)