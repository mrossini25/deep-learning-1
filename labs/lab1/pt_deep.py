from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import data

class PTDeep(nn.Module):
    def __init__(self, config):
        """Arguments:
        - config: list of layer sizes, including input and output layer
        """
        super(PTDeep, self).__init__()
        self.weights = nn.ParameterList([])
        self.biases = nn.ParameterList([])
        for i in range(len(config) - 1):
            self.weights.append(nn.Parameter(torch.randn(config[i], config[i+1])))
            self.biases.append(nn.Parameter(torch.zeros(config[i+1])))
        self.activation = torch.relu
        #self.activation = torch.sigmoid

    def forward(self, X):
        s = torch.mm(X, self.weights[0]) + self.biases[0]
        for i in range(1, len(self.weights)):
            h = self.activation(s)
            s = torch.mm(h, self.weights[i]) + self.biases[i]
        probs = torch.softmax(s, dim=1)
        return probs

    def get_loss(self, X, Yoh_):
        probs = self.forward(X)
        logprobs = torch.log(probs + 1e-10)
        loss = torch.mean(-torch.sum(Yoh_ * logprobs, dim=1))
        return loss
    
    def count_params(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.numel()} parametara, oblik: {param.shape}")

        total = sum(p.numel() for _, p in self.named_parameters())
        print(f"Ukupno parametara: {total}")
    
def train(model, X, Yoh_, param_niter, param_delta, param_lambda = 0):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    # ...
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

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


def ptdeep_decfun(model):
    def classify(X):
        probs = eval(model, X)
        return np.argmax(probs, axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gauss_2d(3, 100)
    #X, Y_ = data.sample_gmm_2d(4, 2, 40)
    #X, Y_ = data.sample_gmm_2d(6, 3, 10)
    Yoh_ = data.class_to_onehot(Y_)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_torch = torch.from_numpy(X).float()
    Y_torch = torch.from_numpy(Yoh_).float()

    # definiraj model:
    ptdeep = PTDeep([2, 3])
    #ptdeep = PTDeep([2, 2])
    #ptdeep = PTDeep([2, 10, 2])
    #ptdeep = PTDeep([2, 10, 10, 2])

    train(ptdeep, X_torch, Y_torch, 10000, 0.1, 0.0001)

    probs = eval(ptdeep, X)

    Y = np.argmax(probs, axis=1)
    accuracy, prec_rec, confmat = data.eval_perf_multi(Y, Y_)
    for i, (recall_i, precision_i) in enumerate(prec_rec):
        print(f"Razred {i}: Precision = {precision_i:.3f}, Recall = {recall_i:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    ptdeep.count_params()

    decfun = ptdeep_decfun(ptdeep)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.title(f'Config: [2, 3], K, C, N = (6, 3, 10)')
    plt.show()


    # testiranje
    np.random.seed(100)
    K = 6
    C = 2
    N = 10
    X, Y_ = data.sample_gmm_2d(K, C, N)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Yoh_ = data.class_to_onehot(Y_)

    X_torch = torch.from_numpy(X).float()
    Y_torch = torch.from_numpy(Yoh_).float()

    konfiguracije = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]
    for config in konfiguracije:
        ptdeep = PTDeep(config)

        train(ptdeep, X_torch, Y_torch, 10000, 0.1, 0.0001)

        probs = eval(ptdeep, X)

        Y = np.argmax(probs, axis=1)
        accuracy, prec_rec, confmat = data.eval_perf_multi(Y, Y_)
        for i, (recall_i, precision_i) in enumerate(prec_rec):
            print(f"Razred {i}: Precision = {precision_i:.3f}, Recall = {recall_i:.3f}")
        print(f"Accuracy: {accuracy:.3f}")

        ptdeep.count_params()

        decfun = ptdeep_decfun(ptdeep)
        bbox = (np.min(X, axis=0), np.max(X, axis=0))
        data.graph_surface(decfun, bbox, offset=0)
        data.graph_data(X, Y_, Y)
        plt.title(f'Config: {config}, gmm_2d({K}, {C}, {N})')
        plt.show()