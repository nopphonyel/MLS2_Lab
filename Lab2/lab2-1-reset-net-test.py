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

trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=4, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

from torch import nn
from PIL import Image
import urllib

resnet18 = models.resnet18(pretrained=True)
resnet18.eval()

# get dog image
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

input_img = Image.open(filename)
input_tensor = preprocess(input_img)
input_batch = input_tensor.unsqueeze(0)

if cuda.is_available():
    input_batch = input_batch.to('cuda:1')
    resnet18.to('cuda:1')

with no_grad():
    output = resnet18(input_batch)

softmax_scores = nn.functional.softmax(output[0], dim=0)

maxval, maxindex = output.max(1)
print('Maximum value', maxval, 'at index', maxindex)

#resnet18.fc = nn.Linear(512, 10)
