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



# MAIN

plt.figure(0)

# Plot sine function, f(x)
t = np.arange(0,1,0.01)
y_sin = np.sin(2*np.pi*t)

# Pick x uniformly in [0, 1]. Do it T times for two sets of inputs.

T = 20
x = np.zeros([T, 2])
for i in range(T):
    x[i, 0] = (random.random())
    x[i, 1] = (random.random())

b = np.sin(2* np.pi*x)

lamb = 1000
# Limit weights with different values of lambda
for i in range(6):
    print("-----------------------------------------------")
    print("Iteration Number: {}".format(i+1))
    plt.figure(0)
    plt.plot(t, y_sin, 'g')

    lamb /= 10

    f2 = []
    w = []
    for i in range(T):
            f2_temp, w_temp = regularization(x[i, :], b[i, :], 1, lamb)
            f2.append(f2_temp)
            w.append(w_temp)


    # Plot all thes estimate on one Plot

    px1 = 0
    px2 = 1
    f2 = np.array(f2)
    w = np.array(w)
    #print("\nWeights:\n{}".format(w))

    for i in range(T):
        plt.plot([0, 1], f2[i], '-k', linewidth = .5)

    # Plot the average estime f1_bar
    f2_bar = [0,0]
    f2_bar[0] = np.mean(f2[:, 0])
    f2_bar[1] = np.mean(f2[:, 1])

    plt.plot([px1, px2], f2_bar, '--', c='r', label= 'first f2 mean')

    f2_1bar = np.mean(f2)

    # Find the bias
    f2_bias = y_sin - f2_1bar

    # Variance
    f2_var = np.mean((b - f2_1bar)**2)

    rms  = np.sqrt(np.mean((y_sin - f2_1bar)**2))

    print(" \nF2 RMS Bias: \n{}".format(rms))
    print(" \nF2 Variance: \n{}".format(f2_var))

    plt.legend()
    plt.show()
