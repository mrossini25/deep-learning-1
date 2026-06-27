import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from zad1 import NLPDataset, Vocab, generate_embedding_matrix, pad_collate_fn
from zad2 import BaselineModel


config = {
    'max_size': -1,
    'min_freq': 1,
    'seed': 13052026,
    'lr': 1e-4,
    'train_batch_size': 10,
    'grad_clip': 0.25,
    'val_batch_size': 32,
    'test_batch_size': 32,
    'epochs': 5,
    'embedding_dim': 300,
    'hidden_size': 150,
    'num_layers': 2,
    'dropout': 0,
    'bidirectional': False,
    'model_type': 'baseline' # Options: 'baseline', 'vanilla_rnn', 'gru', 'lstm'
}


class RNNModel(nn.Module):
    def __init__(self, embedding_layer, hidden_size=150, num_layers=2, dropout=0, bidirectional=False):
        super().__init__()
        self.embedding = embedding_layer
        self.rnn = None
        self.bidirectional = bidirectional

        if bidirectional:
            input_size = hidden_size * 2
        else:
            input_size = hidden_size

        self.decoder = nn.Sequential(
            nn.Linear(input_size, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)

        output, hidden = self.rnn(embedded)
 
        if isinstance(hidden, tuple):
            h = hidden[0]
        else:
            h = hidden

        if self.bidirectional:
            last_hidden = torch.cat((h[-2], h[-1]), dim=1)
        else:
            last_hidden = h[-1]
        
        logits = self.decoder(last_hidden)
        return logits.squeeze(1)
    

class VanillaRNN(RNNModel):
    def __init__(self, embedding_layer, hidden_size=150, num_layers=2, dropout=0, bidirectional=False):
        super().__init__(embedding_layer, hidden_size, num_layers, dropout, bidirectional)
        self.rnn = nn.RNN(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )


class GRU(RNNModel):
    def __init__(self, embedding_layer, hidden_size=150, num_layers=2, dropout=0, bidirectional=False):
        super().__init__(embedding_layer, hidden_size, num_layers, dropout, bidirectional)
        self.rnn = nn.GRU(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )


class LSTM(RNNModel):
    def __init__(self, embedding_layer, hidden_size=150, num_layers=2, dropout=0, bidirectional=False):
        super().__init__(embedding_layer, hidden_size, num_layers, dropout, bidirectional)
        self.rnn = nn.LSTM(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

def choose_model(model_type, config, embedding_layer):
    if model_type == 'baseline':
        return BaselineModel(embedding_layer)
    elif model_type == 'vanilla_rnn':
        return VanillaRNN(embedding_layer, hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=config['bidirectional'])
    elif model_type == 'gru':
        return GRU(embedding_layer, hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=config['bidirectional'])
    elif model_type == 'lstm':
        return LSTM(embedding_layer, hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=config['bidirectional'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(model, data, optimizer, criterion):
    model.train()
    for batch_num, batch in enumerate(data):
        x, y, lengths = batch
        y = y.float()

        model.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
        optimizer.step()


def evaluate(model, data, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, y, lengths = batch
            y = y.float()

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            predicted = (torch.sigmoid(logits) >= 0.5)
            all_preds.extend(predicted.tolist())
            all_labels.extend(y.tolist())

    avg_loss = total_loss / len(data)
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, cm


def main():
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    train_dataset = NLPDataset(os.path.join(BASE_DIR, "sst_train_raw.csv"), 
                                text_vocab=Vocab(max_size=config['max_size'], 
                                                min_freq=config['min_freq']), 
                                label_vocab=Vocab(label=True))
    val_dataset = NLPDataset(os.path.join(BASE_DIR, "sst_valid_raw.csv"),
                        text_vocab=train_dataset.text_vocab,
                        label_vocab=train_dataset.label_vocab)

    test_dataset = NLPDataset(os.path.join(BASE_DIR, "sst_test_raw.csv"),
                            text_vocab=train_dataset.text_vocab,
                            label_vocab=train_dataset.label_vocab)

    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False, collate_fn=pad_collate_fn)

    embedding_layer = generate_embedding_matrix(train_dataset.text_vocab, embedding_dim=config['embedding_dim'], path=os.path.join(BASE_DIR, "sst_glove_6b_300d.txt"))
    model = choose_model(config['model_type'], config, embedding_layer)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    print(f'Config: {config}')

    for epoch in range(config['epochs']):
        train(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, val_cm = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1} / {config['epochs']}: Validation loss: {val_loss:.4f}, accuracy: {val_acc:.3f}%, F1: {val_f1:.4f}")
        # print(f" Confusion matrix:\n{val_cm}")

    test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, criterion)
    print(f"\nTest loss: {test_loss:.4f}, accuracy: {test_acc:.3f}%, F1: {test_f1:.4f}")
    # print(f"Test confusion matrix:\n{test_cm}")


if __name__ == '__main__':
    main()