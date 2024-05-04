import torch.nn as nn


class CNN(nn.Sequential):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(256 * 1 * 1, num_classes)
