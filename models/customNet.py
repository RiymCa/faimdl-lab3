# Define the custom neural network
import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Add more layers...
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 28 * 28, 200) # 200 is the number of classes in TinyImageNet
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.pool(torch.relu(self.conv1(x))) # B x 64 x 224 x 224
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x