import numpy as np #Importing numpy library 
import matplotlib.pyplot as plt #Importing Matplot library for plotting graphs

from Perceptron import Perceptron #Importing the Perceptron class 

#Defining the AND truth table 
inputs = []
inputs.append(np.array([0,0])) #X1 is ON and X2 is OFF
inputs.append(np.array([0,1])) #X1 is OFF and X2 is ON
inputs.append(np.array([1,0])) #X1 is ON and X2 is OFF
inputs.append(np.array([1,1])) #X1 is ON and X2 is ON

#Defining the output for the above truth table 
expected_output = np.array([0,0,0,1])

perceptron = Perceptron(2) #Passing 2 to get the default learning rate and iterations
perceptron.train_model(inputs, expected_output)
#Printing weights in the terminal to view the weights
print(perceptron.weights)
#Defining the graph for the final weights
w = perceptron.weights
m = -w[1]/w[2]
b = -w[0]/w[2]
x = np.array([0,1]) #Defining the boundary of the graph from 0 - 1
y = m*x + b #Linear line equation y=mx+c
fig = plt.figure(1)
x0 = np.array([0,0,1,1]) #Truth table for AND in array 1
x1 = np.array([0,1,0,1]) #Truth table for AND in array 2 
ANDgate= np.array([0,0,0,1]) #Expected output for the truth table
plt.scatter(x0,x1, c=ANDgate)
plt.plot(x,y, label='Final Line', color='blue') #Plot the final line 
plt.legend() #Displaying the legend
plt.legend(loc='lower left', fontsize='small', ncol=2) #Defining the location, size and column of legend
plt.title('Linearly Separable AND Function using Perceptron - CIS(568)') #Title of the graph
plt.show() #Show the graph
