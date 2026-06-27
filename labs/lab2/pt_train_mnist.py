import time
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from funkcije import draw_conv_filters1 as draw_conv_filters
from pt_layers import ConvolutionalModel

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out' / 'pt' / 'MNIST'
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-2
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)

ds_train = MNIST(DATA_DIR, train=True, download=True)

train_x = ds_train.data[:55000].float() / 255.0
train_mean = train_x.mean().item()
train_std = train_x.std().item()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((train_mean,), (train_std,))
    ])

trainset, testset = MNIST(DATA_DIR, train=True, download=True, transform=transform), MNIST(DATA_DIR, train=False, transform=transform)
trainset, validationset = torch.utils.data.random_split(trainset, [55000, 5000])

train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)
validation_loader = DataLoader(validationset, batch_size=config['batch_size'], shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvolutionalModel().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=config['weight_decay'])
criterion = torch.nn.CrossEntropyLoss()

def evaluate(model, loader, criterion):
    model.eval()
    correct, total_loss, total = 0, 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            predicted = torch.argmax(logits.data, 1)
            total_loss += criterion(logits, labels).item()
            correct += (predicted == labels).sum().item()
            total += inputs.size(0)

    acc = correct / total
    loss_avg = total_loss / len(loader)
    return loss_avg, acc

def train(model, train_loader, validation_loader, optimizer, criterion, config):
    lr_policy = config['lr_policy']
    max_epochs = config['max_epochs']
    savedir = config['save_dir']
    batch_size = config['batch_size']
    losses = []
    
    for epoch in range(1, max_epochs + 1):
        if epoch in lr_policy:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['lr_policy'][epoch]['lr']
        model.train()
        total_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, i, len(train_loader), loss))
                draw_conv_filters(epoch, i*batch_size, model.conv1, savedir)
        
        losses.append(total_loss)
        train_loss, train_acc = evaluate(model, train_loader, criterion)
        validation_loss, validation_acc = evaluate(model, validation_loader, criterion)
        print(f'Epoch {epoch}, Train Accuracy: {train_acc}')
        print(f'Epoch {epoch}, Validation Accuracy: {validation_acc}, Validation Loss: {validation_loss}')
    return losses
        

losses = train(model, train_loader, validation_loader, optimizer, criterion, config)
test_loss, accuracy = evaluate(model, test_loader, criterion)
print(f'Test Accuracy: {accuracy}, Test Loss: {test_loss}')

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training loss for weight decay = {config["weight_decay"]}')
plt.show()