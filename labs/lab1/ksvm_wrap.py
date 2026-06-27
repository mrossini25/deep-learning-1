import numpy as np
import torch
from sklearn import svm
import matplotlib.pyplot as plt
import data
import pt_deep

class KSVMWrap:
    '''
    Metode:
    __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre

    predict(self, X)
        Predviđa i vraća indekse razreda podataka X

    get_scores(self, X):
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.

    support
        Indeksi podataka koji su odabrani za potporne vektore
    '''
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.model = svm.SVC(kernel='rbf', C=param_svm_c, gamma=param_svm_gamma)
        self.model.fit(X, Y_)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self, X):
        return self.model.decision_function(X)
    
    def support(self):
        return self.model.support_
    

if __name__=="__main__":
    np.random.seed(100)
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    model = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')
    Y = model.predict(X)
    accuracy, (precision, recall), M = data.eval_perf_multi(Y, Y_)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    decfun = model.get_scores
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0)
    data.graph_data(X, Y_, Y, special=model.support())
    plt.show()


    np.random.seed(100)
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    Yoh_ = data.class_to_onehot(Y_)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_torch = torch.from_numpy(X).float()
    Y_torch = torch.from_numpy(Yoh_).float()

    # definiraj model:
    ptdeep = pt_deep.PTDeep([2, 10, 2])
    #ptdeep = pt_deep.PTDeep([2, 2])
    #ptdeep = pt_deep.PTDeep([2, 10, 2])
    #ptdeep = pt_deep.PTDeep([2, 10, 10, 2])

    pt_deep.train(ptdeep, X_torch, Y_torch, 10000, 0.1, 0.0001)

    probs = pt_deep.eval(ptdeep, X)

    Y = np.argmax(probs, axis=1)
    accuracy, prec_rec, confmat = data.eval_perf_multi(Y, Y_)
    for i, (recall_i, precision_i) in enumerate(prec_rec):
        print(f"Razred {i}: Precision = {precision_i:.3f}, Recall = {recall_i:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    ptdeep.count_params()

    decfun = pt_deep.ptdeep_decfun(ptdeep)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.show()