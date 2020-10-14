import matplotlib.pyplot as plt
import numpy as np
import random

# Plot sine function, f(x)
t = np.arange(0,1,0.01)
y_sin = np.sin(2*np.pi*t)
plt.plot(t, y_sin, 'g')

# Pick x uniformly in [0, 1]. Do it T times.

T = 20
x = np.zeros([T, 2])
for i in range(T):
    x[i][0] = (random.random())
    x[i][1] = (random.random())

x = np.array(x)
b = np.sin(2* np.pi*x)

# For our estimate f_hat1, take the average height
f1 =  x* + ((np.sum(b, axis = 1))/2)

# Plot all thes estimate on one Plot

px1 = 0
px2 = 1

for i in range(T):
    plt.plot([px1, px2], [f1[i], f1[i]], '-k')

# Plot the average estime f1_bar
f1_bar = np.mean(f1)

plt.plot([px1, px2], [f1_bar, f1_bar], 'r' )

# Find the bias
f1_bias = y_sin - f1_bar
# This is a function of t
# Find RMS
f1_rms_bias = np.sqrt(np.mean(f1_bias**2))

# Square this for bias^2 in the book ex 2.8
# Variance
f1_var = np.mean((f1_bar)**2)

plt.show()
