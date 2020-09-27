import numpy as np #Importing numpy library 
import matplotlib.pyplot as plt #Importing Matplot library for plotting graphs

from PocketAlgorithm import Perceptron #Importing the Perceptron class 

#Defining the XOR truth table 
inputs = []
inputs.append(np.array([0,0])) #X1 is ON and X2 is OFF
inputs.append(np.array([0,1])) #X1 is OFF and X2 is ON
inputs.append(np.array([1,0])) #X1 is ON and X2 is OFF
inputs.append(np.array([1,1])) #X1 is ON and X2 is ON

#Defining the output for the above truth table 
expected_output = np.array([0,1,1,0])

perceptron = Perceptron(2, 0.01, 680) #Passing 2 to get the default learning rate and iterations
perceptron.train_model(inputs, expected_output)

inputs = np.array([1, 0]) #To predict data in train model function
perceptron.prediction_method(inputs) 
    
inputs = np.array([0, 1]) #To predict data in train model function
perceptron.prediction_method(inputs)
#Printing weights in the terminal to view the weights
print("Finalized weights: {}\n".format(perceptron.weights))
print("Idealized weights: {}\n".format(perceptron.ideal_weights))
print("Error:\n{} \nLabels:\n{} \nPrediction:\n{}\n".format(perceptron.e, perceptron.l, perceptron.p))
print("\nActivation array (wx + b) ideal: {}\n".format(perceptron.activation))
print("\nAcitvation array Last: {}\n".format(perceptron.activation2))
print("Final Error: \n{}\n".format(perceptron.final_error))
print("Pocket Error: \n{}\n".format(perceptron.count_error))
#Defining the graph for the final weights
w = perceptron.ideal_weights
m = -w[1]/w[2]
b = -w[0]/w[2]
x = np.array([0,1]) #Defining the boundary of the graph from 0 - 1
y1 = m*x[0] + b #Linear line equation y=mx+c
y2 = m*x[1] + b
fig = plt.figure(1)
x0 = np.array([0,0,1,1]) #Truth table for XOR in array 1
x1 = np.array([0,1,0,1]) #Truth table for XOR in array 2 
XORgate= np.array([0,1,1,0]) #Expected output for the truth table
print("(x1, y1): {}\n(x2, y2): {}\n".format((x[0], y1), (x[1], y2)))
plt.scatter(x0,x1, c=XORgate)
plt.plot(x,[y1, y2], label='Final Line', color='blue') #Plot the final line 
plt.legend() #Displaying the legend
plt.legend(loc='lower left', fontsize='small', ncol=2) #Defining the location, size and column of legend
plt.title('Pocket Algorithm XOR - CIS(568)') #Title of the graph
#plt.axis([-0.2, 1.2, -0.2, 1.2])
plt.show() #Show the graph
