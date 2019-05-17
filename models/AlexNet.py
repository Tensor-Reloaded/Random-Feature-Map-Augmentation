import torch.nn as nn
from torchvision import transforms
'''
modified to fit dataset size
'''
NUM_CLASSES = 10

class DivByMax(nn.Module):
    def __init__(self, dim, pow=1):
        super().__init__()
        self.pow = pow
        self.dim = dim

    def forward(self, x):
        x = x ** self.po
        sums = x.sum(dim=self.dim)
        x = x / (sums + 1e-20)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.initial_bn = nn.BatchNorm2d(3)
        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
            # nn.Softmax(),
        )

    def forward(self, x):
        x = self.initial_bn(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
