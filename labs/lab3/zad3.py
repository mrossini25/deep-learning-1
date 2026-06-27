import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from zad1 import NLPDataset, Vocab, generate_embedding_matrix, pad_collate_fn

config = {
    'max_size': -1,
    'min_freq': 1,
    'seed': 42,
    'lr': 1e-4,
    'train_batch_size': 10,
    'grad_clip': 0.25,
    'val_batch_size': 32,
    'test_batch_size': 32,
    'epochs': 5
}

class RNNModel(nn.Module):
    def __init__(self, embedding_layer, hidden_size=150, num_layers=2):
        super().__init__()
        self.embedding = embedding_layer
        self.rnn = nn.GRU(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)

        output, hidden = self.rnn(embedded)
 
        last_hidden = hidden[-1]
        
        logits = self.decoder(last_hidden)
        return logits.squeeze(1)


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

    embedding_layer = generate_embedding_matrix(train_dataset.text_vocab, path=os.path.join(BASE_DIR, "sst_glove_6b_300d.txt"))
    model = RNNModel(embedding_layer)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        train(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, val_cm = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch}:")
        print(f" Validation loss: {val_loss:.4f}, accuracy: {val_acc:.3f}%, F1: {val_f1:.4f}")
        print(f" Confusion matrix:\n{val_cm}")

    test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, criterion)
    print(f"\nTest loss: {test_loss:.4f}, accuracy: {test_acc:.3f}%, F1: {test_f1:.4f}")
    print(f"Test confusion matrix:\n{test_cm}")


if __name__ == '__main__':
    main()