import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1) Convolution + Pooling block
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        
        # 2) Pooling layer 
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 3) Fully connected
        self.fc1 = nn.Linear(10816, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

        # Optional: Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # input shape: [batch_size, 3, 60, 60]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = Net()