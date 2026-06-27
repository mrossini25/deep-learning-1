import torch
import torch.nn as nn
import torch.optim as optim


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2])
Y = torch.tensor([3, 5])

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(100):
    # afin regresijski model
    Y_ = a*X + b

    diff = (Y-Y_)

    # kvadratni gubitak
    loss = torch.mean(diff**2)

    # računanje gradijenata
    loss.backward()

    grad_a_manual = -2 * torch.mean(X * diff)
    grad_b_manual = -2 * torch.mean(diff)  

    grad_a_torch = a.grad.item()
    grad_b_torch = b.grad.item()

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
    print(f'manual: grad_a={grad_a_manual}, grad_b={grad_b_manual}')
    print(f'torch:  grad_a={grad_a_torch}, grad_b={grad_b_torch}')


    # korak optimizacije
    optimizer.step()

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()
