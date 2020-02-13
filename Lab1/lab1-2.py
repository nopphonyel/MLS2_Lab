import torch
# Import training data and stuff
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import urllib
import os
from math import floor, ceil

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
triansize, valsize = floor(len(trainset) * 0.8), ceil(len(trainset) * 0.2)
trainset, valset = torch.utils.data.random_split(trainset, [triansize, valsize])

tr_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(valset, batch_size=16, shuffle=True, num_workers=2)

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

# Let's import some optimizer and loss function....?
import torch.optim as optim
import torch.nn as nn

lr = 0.002
momentum = 0.9
print("[INFO] Param : LR=%s, Momentum=%s" % (lr, momentum))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# Before start training, let's find GPU device.
def find_gpu():
    if torch.cuda.is_available():
        print("[INFO] : GPU Found. Using CUDA as preferred device.")
        model.to("cuda:1")
        return "cuda:1"
    else:
        print("[WARN] : GPU Not found ;_;")
        model.to("cpu")
        return "cpu"


# Finding acc
def find_acc(data_loader, tag):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to("cuda:1"), data[1].to("cuda:1")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(' %d' % ( 100.0 * float(correct) / float(total)), end=",")


def train(prefer_dev):
    print("tr_loss, tr_acc, val_loss, val_acc")
    for epoch in range(50):
        loss_val = 0.0
        final_i = 0
        for i, data in enumerate(tr_loader, 0):
            inputs, label = data[0].to(prefer_dev), data[1].to(prefer_dev)

            # Solving for zero?
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            final_i = i

        print(' %.3f' %
              (loss_val / final_i), end=',')

        find_acc(tr_loader, "train_data")

        # Find the validation loss
        loss_val = 0.0
        final_i = 0
        for i, data in enumerate(val_loader, 0):
            inputs, label = data[0].to(prefer_dev), data[1].to(prefer_dev)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()

            loss_val += loss.item()
            final_i = i

        print(' %.3f' %
              (loss_val / final_i), end=",")

        find_acc(val_loader, "va_data")
        print("")

    print('[INFO] Finished Training')


train(find_gpu())
print("Test acc : " , end='')
find_acc(test_loader, "Test set")

# PATH = './cifar_alxnet.pth'
# torch.save(model.state_dict(), PATH)
