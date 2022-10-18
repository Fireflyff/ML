import torch

X = torch.ones(3, 5)
y = X + 3

X_1 = torch.ones(3, 5, requires_grad=True)
y_1 = X_1 + 5

print(X, y)
print(X_1, y_1)

print(X.grad, X_1.grad)
print(y.grad_fn, y_1.grad_fn)

Y_2 = y_1 + 3
Y_3 = Y_2 * 2
print(Y_2.grad_fn, Y_3.grad_fn)
