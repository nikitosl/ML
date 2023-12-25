import numpy as np

# input
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# output
y = np.where(x > 4.5, 0, 1)
print(y)

# initialization
w0, w1 = 1, 1
lr = 0.01
for i in range(10000):
    
    y_pred = 1 / (1 + np.exp(-x * w1 - w0))
    loss = -(y * np.log(y_pred) + (1 - y) * (np.log(1 - y_pred)))

    # dloss/dw = -(y/y_pred - (1-y)/(1-y_pred))*dy_pred/dw
    # dy_pred/dw = y_pred * (1-y_pred) * x
    # dloss/dw = -(y/y_pred - (1-y)/(1-y_pred)) * y_pred * (1-y_pred) * x = (y_pred - y) * x
    
    dldw0 = y_pred - y
    dldw1 = x * (y_pred - y)
    
    w0 -= lr * dldw0.mean()
    w1 -= lr * dldw1.mean()
    
    if i % 500 == 0:
        print(f"{i}: {loss.mean()}")
    
print(w1, w0)
print(1 / (1 + np.exp(-x * w1 - w0)))