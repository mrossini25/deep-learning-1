import torch
from torch import nn
 
class ConvolutionalModel(nn.Module):
  def __init__(self, in_channels = 1, input_size = 28, conv1_width = 16, conv2_width = 32, fc3_width = 512, class_count = 10):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool1 = nn.MaxPool2d(kernel_size=2)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=conv1_width, out_channels=conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    self.relu2 = nn.ReLU()
    self.flatten3 = nn.Flatten()
    feature_size = input_size // 4
    self.fc3 = nn.Linear(in_features=conv2_width*feature_size*feature_size, out_features=fc3_width, bias=True)
    self.relu3 = nn.ReLU()
    self.fc_logits = nn.Linear(in_features=fc3_width, out_features=class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.pool1(h)
    h = self.relu1(h)
    h = self.conv2(h)
    h = self.pool2(h)
    h = self.relu2(h)
    h = self.flatten3(h)
    h = self.fc3(h)
    h = self.relu3(h)
    logits = self.fc_logits(h)
    return logits
  

class SimpleModel(nn.Module):
  def __init__(self, in_channels = 3, input_size = 32, class_count = 10):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=2)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.flatten3 = nn.Flatten()
    self.fc3 = nn.Linear(in_features=32*7*7, out_features=256)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(in_features=256, out_features=128)
    self.relu4 = nn.ReLU()
    self.fc_logits = nn.Linear(in_features=128, out_features=class_count)

  def forward(self, x):
    h = self.conv1(x)
    h = self.relu1(h)
    h = self.pool1(h)
    h = self.conv2(h)
    h = self.relu2(h)
    h = self.pool2(h)
    h = self.flatten3(h)
    h = self.fc3(h)
    h = self.relu3(h)
    h = self.fc4(h)
    h = self.relu4(h)
    logits = self.fc_logits(h)
    return logits