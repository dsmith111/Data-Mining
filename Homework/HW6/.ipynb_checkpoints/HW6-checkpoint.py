# %% codecell
import matplotlib.pyplot as plt
import numpy as np
import random


def design_matrix(order, x):

    A= np.zeros([len(x), order])
    for i in range(order):
        A[:,i] = x**i

    return A


def pseudo_inverse(x, b, order):
    order += 1
    A = design_matrix(order, x)

    w = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,b))
    y = np.dot(A, w)
    return (w, y)



# MAIN

plt.figure(0)

# Plot sine function, f(x)
t = np.arange(0,1,0.01)
y_sin = np.sin(2*np.pi*t)
plt.plot(t, y_sin, 'g', label = "Sine Curve")

# Pick x uniformly in [0, 1]. Do it T times for two sets of inputs.

T = 20
x = np.zeros([T, 2])
for i in range(T):
    x[i, 0] = (random.random())
    x[i, 1] = (random.random())

b = np.sin(2* np.pi*x)

# For our estimate f_hat1, take the average height

#w1, f2_1, A1 = pseudo_inverse(x[:, 0], b[:, 1], 1)
#w2, f2_2, A2 = pseudo_inverse(x[:, 1], b[:, 0], 1)
#f hat and weights will be  Tx2 matrices

f2 = []
w = []

for i in range(T):
        f2_temp, w_temp = pseudo_inverse(x[i, :], b[i, :], 1)
        f2.append(f2_temp)
        w.append(w_temp)


# Plot all thes estimate on one Plot

px1 = 0
px2 = 1
f2 = np.array(f2)
w = np.array(w)

# line that passes through both points

for i in range(T):
    plt.figure(0)
    plt.plot([0, 1], f2[i], '-k', linewidth = .5)
    plt.figure(1)
    plt.plot(x[i], f2[i], '-k', linewidth = .5)


# Plot the average estimate f1_bar
f2_1bar = [0,0]
f2_1bar[0] = np.mean(f2[:, 0])
f2_1bar[1] = np.mean(f2[:, 1])

plt.figure(0)
plt.plot([px1, px2], f2_1bar, '--', c='r', label= 'F2 mean')
plt.legend()
plt.figure(1)
plt.plot(t, y_sin, 'g', label = "Sine Curve")
plt.plot([px1, px2], f2_1bar, '--', c='r', label= 'F2 mean')
plt.legend()

# Find the bias
f2_bar = np.mean(f2)
f2_bias = y_sin - f2_bar

f2_rms = np.sqrt(np.mean(f2_bias**2))
# Variance
f2_var = np.mean((b - f2_bar)**2)

print(" \nF2 RMS: \n{}".format(f2_rms))
print(" \nF2 Variance: \n{}".format(f2_var))

#plt.axis([0, 1, -2, 2])
plt.legend()
plt.show()
