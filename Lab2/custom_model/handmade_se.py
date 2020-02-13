import torch
import torch.nn as nn


class HandmadeSE(nn.Module):
    ratio = 16

    def __init__(self, num_classes=10):
        super(HandmadeSE, self).__init__()
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

        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

        self.FC_r = nn.Sequential(
            nn.Linear(1 * 1 * 384, 384 // self.ratio),
            nn.ReLU(inplace=True),
            nn.Linear(384 // self.ratio, 384),
            nn.Sigmoid()
        )

        self.FC_r2 = nn.Sequential(
            nn.Linear(1 * 1 * 256, 256 // self.ratio),
            nn.ReLU(inplace=True),
            nn.Linear(256 // self.ratio, 256),
            nn.Sigmoid()
        )

        self.project = nn.Conv2d(64, 384, kernel_size=1,
                                 stride=2)  # Seems conv2d only affects the dept, stride param affects the width and height
        self.project2 = nn.Conv2d(384, 256, kernel_size=1)

    arch = 0

    def set_arch(self,arch):
        self.arch = arch

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)

        # SE Part
        if self.arch == 1:
            for_se_path = x
            for_se_path = self.squeeze(for_se_path)
            for_se_path = torch.flatten(for_se_path, 1)
            for_se_path = self.FC_r(for_se_path)
            for_se_path = for_se_path.unsqueeze(-1).unsqueeze(-1)
            x = x * for_se_path

        x = self.block1_act(x)
        x = self.block2(x)

        # SE Part
        if self.arch == 2:
            for_se_path = x
            for_se_path = self.squeeze(for_se_path)
            for_se_path = torch.flatten(for_se_path, 1)
            for_se_path = self.FC_r2(for_se_path)
            for_se_path = for_se_path.unsqueeze(-1).unsqueeze(-1)
            x = x * for_se_path

        x = self.block2_act(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def handmade_se(**kwargs):
    model = HandmadeSE(**kwargs)
    return model
