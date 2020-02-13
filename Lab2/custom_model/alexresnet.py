import torch
import torch.nn as nn


class AlexResNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
        )
        self.block1_act = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )
        self.block2_act = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.project = nn.Conv2d(64, 384, kernel_size=1, stride=2) #Seems conv2d only affects the dept, stride param affects the width and height
        self.project2 = nn.Conv2d(384, 256, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        #print("conv1(x) : " , x.size())
        identity = x[:, :, 1:, 1:] #Eliminate one row and one column to make dimension to 13x13
        #print("identity : ", identity.size())
        x = self.block1(x)
        #print("block1(x) : ", x.size())
        identity = self.project(identity)
        #print("project(iden) : " , identity.size())
        x += identity
        #print("Sum-up : ", x.size())
        x = self.block1_act(x)

        identity2 = x
        x = self.block2(x)
        identity2 = self.project2(identity2)
        #print("block2(x) : ", x.size())
        x += identity2
        x = self.block2_act(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexresnet(**kwargs):
    model = AlexResNet(**kwargs)
    return model
