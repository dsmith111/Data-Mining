# %% codecell
import matplotlib.pyplot as plt
import numpy as np
import random


def design_matrix(order, x):

    A= np.zeros([len(x), order])
    for i in range(order):
        A[:,i] = x**i

    return A

def regularization(x, b, order, lamb):
    order += 1
    A = design_matrix(order, x)

    w = np.linalg.inv(A.T.dot(A) + (lamb*np.identity(np.shape(A)[1]))).dot(A.T.dot(b))
    y = np.dot(A, w)
    return (w, y)



# Problem 1

# Suppose we observe 11 data points given by:
x = np.linspace(-0.5, 0.5, 11)
x = np.array([x])
y_sin = np.sin(2*np.pi*x)
b = y_sin - 0.1 + 0.2*np.random.rand(1,11)
order = 9
lamb = 1000


plt.figure(1)
plt.plot(y_sin, x, color = 'blue')
plt.scatter(x, b, c = 'g')

# Find lambda, which gives the best performance in terms of cross...
# validation error for a ninth-order polynomial model. Use leave-on-out...
# cross validation

#w, f = regularization(x, b,)


plt.show()
