from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from pt_layers import SimpleModel, ConvolutionalModel
from load_cifar import load_cifar
from torchvision.datasets import MNIST

# DATA_DIR = Path(__file__).parent / 'datasets' / 'CIFAR10'
# DATA_DIR.mkdir(parents=True, exist_ok=True)
# train_dataset, validation_dataset, test_dataset, data_mean, data_std = load_cifar(DATA_DIR)
# train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
#model = SimpleModel()

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
DATA_DIR.mkdir(parents=True, exist_ok=True)
trainset = MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(trainset, batch_size=50, shuffle=True)
model = ConvolutionalModel()
#model.forward()

for i, (inputs, labels) in enumerate(train_loader):
    model.forward(inputs)
    if i > 0:
        break 