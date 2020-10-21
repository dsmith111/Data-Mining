import matplotlib.pyplot as plt
import numpy as np
import random
from __future__ import division

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


def validation_error(errors):
    e_val = (1/len(errors)) * sum(errors)
    return e_val

def numpy_pop(array, index):
    array = array.tolist()
    number = array.pop(index)
    array = np.asarray(array)

    return [array, number]


# Leave-one-out cross-validation
def one_cross_validation(x, target, order, lamb, N):

    cv_array = []
    for i in range(len(x)):
        tempx = x[:]
        tempb = target[:]
        tempx, cvx = numpy_pop(tempx, i)
        tempb, dump = numpy_pop(tempb, i)
        tempW, tempf = regularization(tempx, tempb, order, lamb)
        tempA = design_matrix(order+1, np.array([cvx]))
        guess = tempA.dot(tempW)

        e_val = (guess[0]-cvx)**2
        cv_array.append(e_val)

    e_val_sum = validation_error(cv_array)
    return e_val_sum




# Problem 1

# Suppose we observe 11 data points given by:
size = 11
x = np.linspace(-0.5, 0.5, size)
#x = np.array([x])
y_sin = np.sin(2*np.pi*x)
b = y_sin - 0.1 + 0.2*np.random.rand(1,11)
order = 9
b = b[0]
lamb = 10000
cv_history = []
iterations = 100

# Find lambda, which gives the best performance in terms of cross...
# validation error for a ninth-order polynomial model. Use leave-on-out...
# cross validation
for i in range(iterations):

    w, f = regularization(x, b, order, lamb)
    cv = one_cross_validation(x, b, order, lamb, size)
    cv_history.append(cv)
#    if(i%1 != 0):
#        continue

    #Plot
    if cv <= min(cv_history) or i == (iterations - 1):
        plt.figure(i)
        plt.plot(x, y_sin, color = 'green',linestyle = '--' , label = "Sine Curve")
        plt.plot(x, b, c = 'red', label = "Noisy Sine Curve")
        plt.scatter(x, f, color = "cyan", marker = '*', label = "Regularized 9th Order Model")
        plt.title("Iteration {}".format(i+1))
        plt.legend()
        plt.show()
        print("\nLambda {}: \n{}".format(i+1, lamb))
        print("\nCross-validation Error {}: \n{}".format(i+1, cv))
        print("______________________________________________________")
    lamb *= .75


plt.plot(range(iterations), cv_history)
plt.title("Cross-validation History")
plt.show()
