from matplotlib import pyplot as plt
import numpy as np
import data

def fcann2_train(X, Y_):

    param_niter = 100000
    param_delta = 0.05
    param_lambda = 0.001
    hidden_layer_dimension = 5

    w1 = np.random.randn(X.shape[1], hidden_layer_dimension) # (D, H)
    b1 = np.zeros((1, hidden_layer_dimension)) # (1, H)
    w2 = np.random.randn(hidden_layer_dimension, Y_.shape[1]) # (H, C)
    b2 = np.zeros((1, Y_.shape[1])) # (1, C)
    # print(f"Initial W1 norm: {np.linalg.norm(w1):.6f}")
    # print(f"Initial W2 norm: {np.linalg.norm(w2):.6f}")

    for i in range(param_niter):
        # forward pass
        s1 = np.dot(X, w1) + b1 # (N, H)
        hidden_layer = np.maximum(0, s1) # (N, H)
        s2 = np.dot(hidden_layer, w2) + b2 # (N, C)
        expscores = np.exp(s2)
        sumexp = np.sum(expscores, axis=1, keepdims=True)
        probs = expscores / sumexp
        logprobs = np.log(probs)

        loss = -np.mean(np.sum(Y_ * logprobs, axis=1)) + 0.5 * param_lambda * (np.sum(w1*w1) + np.sum(w2*w2))

        if i % 1000 == 0:
            print(f"iteration {i}: loss {loss}")

        # backward pass
        dL_ds = (probs - Y_) # (N, C)
        dL_dw2 = np.dot(hidden_layer.T, dL_ds) / X.shape[0] + param_lambda * w2 # (H, C)
        dL_db2 = np.mean(dL_ds, axis=0, keepdims=True) # (1, C)
        dL_dhidden = np.dot(dL_ds, w2.T) # (N, H)
        dL_ds = dL_dhidden * (s1 > 0)  # (N, H)
        dL_dw1 = np.dot(X.T, dL_ds) / X.shape[0] + param_lambda * w1 # (D, H)
        dL_db1 = np.mean(dL_ds, axis=0, keepdims=True) # (1, H)
        # if i % 1000 == 0 and i < 10000:
        #     print(f"grad_W1 norm: {np.linalg.norm(dL_dw1):.6f}")
        #     print(f"grad_W2 norm: {np.linalg.norm(dL_dw2):.6f}")
        #     print(f"grad_b1 norm: {np.linalg.norm(dL_db1):.6f}")
        #     print(f"grad_b2 norm: {np.linalg.norm(dL_db2):.6f}")

        # parameter update
        w1 += -param_delta * dL_dw1
        b1 += -param_delta * dL_db1
        w2 += -param_delta * dL_dw2
        b2 += -param_delta * dL_db2

    return w1, w2, b1, b2

def fcann2_classify(X, w1, w2, b1, b2):
    s1 = np.dot(X, w1) + b1
    hidden_layer = np.maximum(0, s1)
    s2 = np.dot(hidden_layer, w2) + b2
    expscores = np.exp(s2 - np.max(s2, axis=1, keepdims=True))
    return expscores / np.sum(expscores, axis=1, keepdims=True)  # (N, C)

def fcann2_decfun(w1, w2, b1, b2):
    def classify(X):
        probs = fcann2_classify(X, w1, w2, b1, b2)
        return np.argmax(probs, axis=1)
    return classify

if __name__=="__main__":
    np.random.seed(100)
    
    # get data
    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    Y_oh = data.class_to_onehot(Y_)
    # print(X.shape)
    # print(Y_oh.shape)
    
    # train
    w1, w2, b1, b2 = fcann2_train(X, Y_oh)
    
    # classify
    probs = fcann2_classify(X, w1, w2, b1, b2)
    Y = np.argmax(probs, axis=1)

    accuracy, precision, recall = data.eval_perf_multi(Y, Y_)
    print(f"Accuracy={accuracy:.3f}")

    decfun = fcann2_decfun(w1, w2, b1, b2)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0)
    data.graph_data(X, Y_, Y)
    plt.show()