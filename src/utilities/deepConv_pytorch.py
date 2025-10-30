import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=256, dropoutRate=0.5):
        super(DeepConvNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        self.classifier = nn.Linear(200 * ((Samples // 16) - 1), nb_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)
