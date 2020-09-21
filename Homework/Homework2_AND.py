import matplotlib.pyplot as plt
import numpy as np
import random
import copy


# Method which uses the random library to return a weight value between -1 and 1.
def initialize_weights():
    # Initialize Weights
    w = []

    for i in range(3):
        w.append(random.random()*2 - 1)
        
    return w

# The visualize guess method simply plots the data points, the target line and the guess line.
def visualize_guess(w,finished):

    #Data to Classify
    if finished:
        color = "r"
    else:
        color = (random.random(), random.random(), random.random())
    plt.axis([-1.05, 1.05, -1.05, 1.05])
    plt.plot(0, 0, 'r*')
    plt.plot(1, 0, 'r*')
    plt.plot(0, 1, 'r*')
    plt.plot(1, 1, 'go')

    # Target Decision Line
    # w0 < 0
    # w1 < -w0
    # w2 < -w0
    # w1 + w2 >= -w0
    
    px1 = -1
    m = -1/1 # -w1/w2
    b = 1.1/1 # -w0/w2
    py1 = m*px1 + b
    px2 = 1
    py2 = m*px2 + b

    plt.plot([px1, px2], [py1, py2], 'b')


    # Guess
    m = -w[1]/w[2]
    b = -w[0]/w[2]
    py1 = m*px1 + b
    px2 = 1
    py2 = m*px2 + b

    plt.plot([px1, px2], [py1, py2], color)
    if finished:
        print("x1,y1: {}  | x2,y2  {}".format([px1, py1], [px2, py2]))
    

# The truth table method requires manual input of the x values. It is simply here to display the truth table to the user.
def truth_table(y):

    #Truth Table

    x0 = np.array([(1, 1, 1, 1)])
    x1 = np.array([(1, 1, 0, 0)])
    x2 = np.array([(1, 0, 1, 0)])
    y = np.array([(y)])
    truth = np.concatenate((x1.T, x2.T, y.T), axis = 1)
    print("Truth Table for AND:\n {}\n".format(truth))
    return [x0, x1, x2]


# The activation method returns the activation array of the input weights.
def activation_function(w, x0, x1, x2):
    # Weight Check to Truth Table

    x = np.concatenate((x0.T, x1.T, x2.T),axis = 1)
    g = np.inner(np.array(w),x)
  
    return [x, g]


# The alter weights method decreases or increases the weights by the learning array dependent on which points were activated or not.
def alter_weights(learning_rate,w,x, index, sign):

    # Multiply the x inputs by the learning rate to generate a learning array.
    learning_array = np.multiply(x,learning_rate)
    
    # If the sign of the incorrect activation element is supposed to be positive: increase, if it is supposed to be negative: decrease.
    if sign >= 0:
        w = np.add(np.array(w),learning_array[index])
    elif sign < 0:
        w = np.subtract(np.array(w),learning_array[index])
    return w
    
# Method which takes the "logic array", an array telling if an element is on or off based on its sign, and compares the current with the guess
# The method returns the index of an incorrect node and the sign it is supposed to be.
def find_mismatch(y, logic_array_target, logic_array_guess,random_guess):
    temporary_array = list(range(len(logic_array_guess)))
    
    for i in range(len(logic_array_guess)):

        if random_guess == True:
            selected_index = random.choice(temporary_array)
        else:
            selected_index = i

        if logic_array_guess[selected_index] != logic_array_target[selected_index]:
            
            return [selected_index, logic_array_target[selected_index]]
        
        if len(temporary_array) > 0:
            temporary_array.remove(selected_index)
        

# This is a reduced version of the pocket algorithm. It continuously cycles the weight values producing increasingly accurate...
# values until it is able to correctly separate the nodes. It does not take into account the amount of error. 
def pocket_algorithm(w, g, logic_array_target, learning_rate, x, x_list, y):
    count = 0
    pocket_w = w
    random_guess = False
    count_error = 1
    logic_array_guess = None
    while logic_array_guess != logic_array_target:
        
        count += 1
        logic_array_guess = []
        logic_array_guess = list(map(lambda x: np.sign(x), g))
        
        if (logic_array_guess == logic_array_target):
            break
        
        temp = find_mismatch(y, logic_array_target, logic_array_guess, random_guess)
            
        if temp == None:
            continue
                
        truth_table_index = temp[0]
        sign = temp[1]
        pocket_w = alter_weights(learning_rate, pocket_w, x, truth_table_index, sign)
        g = activation_function(pocket_w, x_list[0], x_list[1], x_list[2])[1]
        
        if count >= 1000:
            print("Took too long.\n")
            break
        
        
    print("Iterations: {}\n".format(count))   
    print(g)
    return pocket_w






# Execution Code Begins Here

## This sets up our variables, prints our initial weights, runs the pocket algorithm, then prints the final weights and displays the chart.
learning_rate = 0.01
logic_array_target = [1, -1, -1, -1]
output = [1, 0, 0, 0]
w = initialize_weights()

print("Weights: {}\n".format(w))

x_list = truth_table(output)
[x, g] = activation_function(w, x_list[0], x_list[1], x_list[2])
w = pocket_algorithm(w, g, logic_array_target, learning_rate, x, x_list, output)

print("Finalized Weights: {}\n".format(w))

visualize_guess(w,True)
plt.show()