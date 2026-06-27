import time
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from funkcije import draw_conv_filters2 as draw_conv_filters, plot_training_progress
from load_cifar import load_cifar
from pt_layers import SimpleModel


DATA_DIR = Path(__file__).parent / 'datasets' / 'CIFAR10'
SAVE_DIR = Path(__file__).parent / 'out' / 'pt' / 'CIFAR10'
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(int(time.time() * 1e6) % 2**31)

config = {}
config['max_epochs'] = 50
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-4
config['lr_policy'] = 1e-2

train_dataset, validation_dataset, test_dataset, data_mean, data_std = load_cifar(DATA_DIR)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=config['lr_policy'], weight_decay=config['weight_decay'])
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


def show_worst_misclassified(inputs, labels, preds, losses, probs, num_images=20):
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    misclassified = np.where(preds != labels)[0]
    misclassified = misclassified[np.argsort(losses[misclassified])[::-1][:num_images]]
    
    for plot_idx, idx in enumerate(misclassified):
        img = inputs[idx].transpose(1, 2, 0) * data_std + data_mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        top3_idx  = np.argsort(probs[idx])[::-1][:3]
        top3_prob = probs[idx][top3_idx]
        title = f'True: {labels[idx]}, Pred: {preds[idx]}\n'
        title += 'Top 3: ' + ', '.join([f'{i}:{p:.2f}' for i, p in zip(top3_idx, top3_prob)])
        axes[plot_idx].imshow(img)
        axes[plot_idx].axis('off')
        axes[plot_idx].set_title(title)

    fig.tight_layout()
    plt.savefig(Path(SAVE_DIR) / 'worst_misclassified.png')
    plt.show()

def evaluate(model, loader, criterion, prikazi_pogresne_slike=False):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    all_losses = []
    all_inputs = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            predicted = torch.argmax(logits.data, 1)
            total_loss += criterion(logits, labels).item()
            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())

            if prikazi_pogresne_slike:
                per_sample_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                all_losses.append(per_sample_loss.cpu())
                all_inputs.append(inputs.cpu())
                all_probs.append(torch.softmax(logits, dim=1).cpu())

    all_preds  = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    num_classes = max(all_labels) + 1

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(all_labels, all_preds):
        confusion_matrix[true][pred] += 1

    TP = np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - TP
    FN = confusion_matrix.sum(axis=1) - TP

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    accuracy  = TP.sum() / confusion_matrix.sum()
    loss_avg = total_loss / len(loader)

    if prikazi_pogresne_slike:
        all_losses = torch.cat(all_losses).numpy()
        all_inputs = torch.cat(all_inputs).numpy()
        all_probs  = torch.cat(all_probs).numpy()
        show_worst_misclassified(all_inputs, all_labels, all_preds, all_losses, all_probs)

    return loss_avg, accuracy, confusion_matrix

def train(model, train_loader, validation_loader, optimizer, criterion, scheduler, config):
    max_epochs = config['max_epochs']
    savedir    = config['save_dir']
    
    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    for epoch in range(1, max_epochs + 1):
        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, i, len(train_loader), loss))

            if i % 200 == 0:
                draw_conv_filters(epoch, i, model.conv1.weight.detach().cpu().numpy(), savedir)

        train_loss, train_acc, train_conf_mat = evaluate(model, train_loader, criterion)
        validation_loss, validation_acc, validation_conf_mat = evaluate(model, validation_loader, criterion)

        print(f'Epoch {epoch}, Train Accuracy: {train_acc}')
        print(f'Epoch {epoch}, Validation Accuracy: {validation_acc}, Validation Loss: {validation_loss}')
        print(f'Validation Confusion Matrix:\n{validation_conf_mat}')

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [validation_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [validation_acc]
        plot_data['lr'] += [scheduler.get_last_lr()]
        scheduler.step()

    return plot_data


plot_data = train(model, train_loader, validation_loader, optimizer, criterion, scheduler, config)

test_loss, accuracy, test_conf_mat = evaluate(model, test_loader, criterion, prikazi_pogresne_slike=True)
print(f'Test Accuracy: {accuracy}, Test Loss: {test_loss}')
print(f'Test Confusion Matrix:\n{test_conf_mat}')

plot_training_progress(SAVE_DIR, plot_data)