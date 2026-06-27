from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import data

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
        - D: dimensions of each datapoint 
        - C: number of classes
        """

        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        super(PTLogreg, self).__init__()
        self.W = nn.Parameter(torch.randn(D, C))
        self.b = nn.Parameter(torch.zeros(C))
        self.weight_decay = 0.01

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...
        logits = torch.mm(X, self.W) + self.b
        probs = torch.softmax(logits, dim=1)
        return probs

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.exp, torch.sum
        #   pripaziti na numerički preljev i podljev
        # ...
        probs = self.forward(X)
        logprobs = torch.log(probs + 1e-10)
        return torch.mean(-torch.sum(Yoh_ * logprobs, dim=1)) + 0.5 * self.weight_decay * torch.norm(self.W, p=2)
    

def train(model, X, Yoh_, param_niter, param_delta):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    # ...
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    # ...
    for i in range(param_niter):
        optimizer.zero_grad()
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        if i % 100 == 0:
            print(f'step: {i}, loss:{loss}')
        optimizer.step()


def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    X_torch = torch.from_numpy(X).float()
    with torch.no_grad():
        probs = model.forward(X_torch)
    return probs.detach().numpy()


def logreg_decfun(model):
    def classify(X):
        probs = eval(model, X)
        return np.argmax(probs, axis=1)
    return classify
    
if __name__ == "__main__":
    np.random.seed(100)

    C = 3
    X, Y_ = data.sample_gauss_2d(C, 100)
    #X, Y_ = data.sample_gmm_2d(6, C, 10)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Yoh_ = data.class_to_onehot(Y_)

    X_torch = torch.from_numpy(X).float()
    Y_torch = torch.from_numpy(Yoh_).float()

    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    train(ptlr, X_torch, Y_torch, 1000, 0.5)

    probs = eval(ptlr, X)

    Y = np.argmax(probs, axis=1)
    accuracy, prec_rec, confmat = data.eval_perf_multi(Y, Y_)
    for i, (recall_i, precision_i) in enumerate(prec_rec):
        print(f"Razred {i}: Precision = {precision_i:.3f}, Recall = {recall_i:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    decfun = logreg_decfun(ptlr)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=None)
    data.graph_data(X, Y_, Y)
    plt.show()