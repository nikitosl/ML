import numpy as np

# input
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# output
y = 3 * x + 7

# y_pred = w1 * x + w0
# loss = (y_pred - y) ** 2 (MSE)
# dloss/dw = 2 * (y_pred - y) * dy_pred/dw = 2 * (y - y_pred) * x

w0, w1 = 10, 10

lr = 0.01
for i in range(1000):
    ldw0 = 2 * (w1 * x + w0 - y).mean()
    ldw1 = 2 * (x * (w1 * x + w0 - y)).mean()

    w0 -= lr * ldw0
    w1 -= lr * ldw1

print(w1, w0)