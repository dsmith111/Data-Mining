import numpy as np                              #importing the numpy libraries 
#This library helps in all linear algebra functions like dot product
import matplotlib.pyplot as plt

#Defining Main class that will be called
class Perceptron(object):

    #Defining the constructor with the parameters 
    #Setting default learning rate of 0.2 and 20 iteration
    def __init__(self, no_of_inputs, learning_rate=0.1,no_of_iterations=8): 
        self.no_of_iterations = no_of_iterations #Setting to variable 
        self.learning_rate = learning_rate #Setting to variabe 
        self.weights = np.random.rand(no_of_inputs+1) #Initialization weight vector by using np.random to create vector 

    #Defining the prediction method 
    def prediction_method(self, x):
        sum = np.dot(x, self.weights[1:]) + self.weights[0] #numpy dot product for vectors

        #Understanding that if x > 0 then 1 else 0 
        if sum > 0: #For any number above 0
            activation = 1 #Save value as 1 
        else:
            activation = 0 #Otherwise save 0 
        return activation

    #Defining the training method with parameters
    def train_model(self, inputs, corresponsing_inputs):
        #Reaching our limit with the below for loop
        for i in range (self.no_of_iterations):
            for x, label in zip(inputs, corresponsing_inputs): #New object 
                prediction = self.prediction_method(x) #Passing x into prediction method 
                self.weights[1:] += self.learning_rate * (label - prediction) * x #W = 
                self.weights[0] += self.learning_rate * (label - prediction) #Updating the BIAS
                
                #Plot the graph using matplotlib
                w=self.weights #Assigning weights to w
                m = -w[1]/w[2]
                b = -w[0]/w[2]
                x=np.array([1,0])
                y=m*x+b #Linear line equation y=mx+c
                fig = plt.figure(1)
                x0 = np.array([0,0,1,1]) #Truth table for AND in array 1
                x1 = np.array([0,1,0,1]) #Truth table for AND in array 2 
                ANDgate= np.array([0,0,0,1]) #Expected output for the truth table
                plt.scatter(x0,x1, c=ANDgate)
            plt.plot(x,y, label='Line: ' + str(i)) #Plot the lines