import numpy as np                              #importing the numpy libraries 
#This library helps in all linear algebra functions like dot product
import matplotlib.pyplot as plt
import copy

#Defining Main class that will be called
class Perceptron(object):

    #Defining the constructor with the parameters 
    #Setting default learning rate of 0.01 and 100 iteration
    def __init__(self, no_of_inputs, learning_rate=0.01,no_of_iterations=100): 
        self.no_of_iterations = no_of_iterations #Setting to variable 
        self.learning_rate = learning_rate #Setting to variabe 
        self.weights = np.random.rand(no_of_inputs+1) #Initialization weight vector by using np.random to create vector 
        self.ideal_weights = np.random.rand(no_of_inputs + 1) #Ideal weights for pocket algorithm
        self.count_error = -1 #Variable to count the error
        self.record_error = [] #Array to record the error
        self.count_iterations = 0 # Variable keeping track of number of iterations
        self.final_error = 0
        self.final_weights = np.random.rand(no_of_inputs + 1)
        self.activation = []
        self.activation2 = []
        self.e = []
        self.l = []
        self.p = []
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
            pocket_errors = 0
            
            for x, label in zip(inputs, corresponsing_inputs): #New object 
                prediction = self.prediction_method(x) #Passing x into prediction method 
                error = label - prediction #Defining error
                
                if i == self.no_of_iterations-1:
                    self.e.append(error)
                    self.l.append(label)
                    self.p.append(prediction)
                    
                update_weight = self.learning_rate * error #Updating by multiplaying learning with error
                self.weights[1:] += self.learning_rate * (label - prediction) * x #W = 
                self.weights[0] += self.learning_rate * (label - prediction) #Updating the BIAS
                
                pocket_errors += int(update_weight!= 0.0) #Updating the error
                
            if i == self.no_of_iterations - 1:
                self.final_error = pocket_errors
                self.final_weights = self.weights
                x0 = np.array([(1, 1, 1, 1)])
                x1 = np.array([(1, 1, 0, 0)])
                x2 = np.array([(1, 0, 1, 0)])
                xf = np.concatenate((x0.T, x1.T, x2.T),axis = 1)
                wtX = np.inner(np.array(self.weights), xf)
                self.activation2.append(wtX)                
                
            if(self.count_error == -1) or (self.count_error > pocket_errors) or (pocket_errors == 0):
                self.ideal_weights = copy.deepcopy(self.weights)
                self.count_error = copy.copy(pocket_errors)
                x0 = np.array([(1, 1, 1, 1)])
                x1 = np.array([(1, 1, 0, 0)])
                x2 = np.array([(1, 0, 1, 0)])
                xf = np.concatenate((x0.T, x1.T, x2.T),axis = 1)
                wtX = np.inner(np.array(self.weights), xf)
                self.activation.append(wtX)
            
            if (pocket_errors == 0):
                break
            self.record_error.append(self.count_error)
        
        self.count_iterations = i
        
    def classifySample(self, data):
        result_predicted = np.where((prediction_method(data)+ self.weights[0]) > 0.0, 1, 0)
        return [self._label_map[item] for item in result_predicted]