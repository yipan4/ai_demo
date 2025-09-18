import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)    
        self.fc1 = nn.Linear(64 * 7, 512)
        # self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10) 

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    model = CNN()
    print(model)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print("Output shape:", output.shape)