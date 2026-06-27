import time
import torch.optim
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from model import SimpleMetricEmbedding, IdentityModel
from utils import train, evaluate, compute_representations
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
params_dir = os.path.join(script_dir, "params")
os.makedirs(params_dir, exist_ok=True)

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False

MODEL_TYPE = "simple" # "simple" or "identity"
REMOVE_CLASS = None


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./mnist/"
    if REMOVE_CLASS is not None:
        print(f"Removing class {REMOVE_CLASS} from training set")
        ds_train = MNISTMetricDataset(mnist_download_root, split='train', remove_class=REMOVE_CLASS)
    else:
        ds_train = MNISTMetricDataset(mnist_download_root, split='train')

    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    emb_size = 32
    if MODEL_TYPE == "simple":
        model = SimpleMetricEmbedding(1, emb_size).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

        epochs = 3
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            t0 = time.time_ns()
            train_loss = train(model, optimizer, train_loader, device)
            print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
            if EVAL_ON_TEST or EVAL_ON_TRAIN:
                print("Computing mean representations for evaluation...")
                representations = compute_representations(model, train_loader, num_classes, emb_size, device)
            if EVAL_ON_TRAIN:
                print("Evaluating on training set...")
                acc1 = evaluate(model, representations, traineval_loader, device)
                print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
            if EVAL_ON_TEST:
                print("Evaluating on test set...")
                acc1 = evaluate(model, representations, test_loader, device)
                print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
            t1 = time.time_ns()
            print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}")
        if REMOVE_CLASS is not None:
            torch.save(model.state_dict(), os.path.join(params_dir, f"{type(model).__name__}_acc{acc1:.4f}_removed_class_{REMOVE_CLASS}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(params_dir, f"{type(model).__name__}_acc{acc1:.4f}.pth"))


    else:
        model = IdentityModel().to(device)
        emb_size = 28 * 28
        representations = compute_representations(model, traineval_loader, num_classes, emb_size, device)
        acc1 = evaluate(model, representations, test_loader, device)
        print(f"Test Accuracy (image space): {acc1 * 100:.2f}%")
        torch.save(model.state_dict(), os.path.join(params_dir, f"{type(model).__name__}_acc{acc1:.4f}.pth"))