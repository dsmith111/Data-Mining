import numpy as np
import matplotlib.pyplot as plt


# Least Squares

def lst_square(x, y):

    sx = np.sum(x)
    sxy = np.dot(x,y)
    sxx = np.dot(x,x)
    sy = np.sum(y)
    n = len(x)
    print(sx)
    print(sxx)
    print(sxy)
    print(sy)
    m = ((n*sxy) - (sx*sy))/((n*sxx) - (sx^2))
    b = (sy - (m*sx))/n

    print("m: {}\n".format(m))
    print("b: {}\n".format(b))

    return (m, b)

# R^2
# p-value for R^2
# plot graph

def visualize(y, x, m, b):

    # Plotting points
    
    plt.plot(x, y, 'ro')

    # Plotting Least Squares Line

    px1 = 1980
    py1 = (m*px1) + b
    px2 = 2000
    py2 = (m*px2) + b
    print("y1: {}\n".format(py1))
    print("y2: {}\n".format(py2))
    plt.plot([px1, px2], [py1, py2], 'b')
#
#
#



# Main execution block

x = np.array([1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996, 1999, 2000])
y = np.array([338.7, 341.1, 344.4, 347.2, 351.5, 354.2, 356.4, 358.9, 362.6, 366.6, 369.4])

[m, b] = lst_square(x, y)
visualize(y, x, m, b)

plt.show()