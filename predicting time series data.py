import copy, numpy as np
import time
np.random.seed(0)
n = 8000
row = 1000
col = 8
A = np.array(range(n))/(n-1)
B = np.sin(np.array(range(n)))
C = np.sin(np.array(range(1,n+1)))
A = A.reshape(row,col)
B = B.reshape(row,col)
C = C.reshape(row,col)
B = (B - np.min(B)) / (np.max(B) - np.min(B))
C = (C - np.min(C)) / (np.max(C) - np.min(C))
# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1
binary_dim = 8
# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(1000000):
    index = np.random.randint(row)
    a = A[index,:]
    b = B[index,:]
    c = C[index,:]   
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)
    overallError = 0
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T
        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])  
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = layer_2[0][0]      
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1)) 
    future_layer_1_delta = np.zeros(hidden_dim)
    for position in range(binary_dim):    
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]    
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)        
        future_layer_1_delta = layer_1_delta
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0  
    # print out progress
    if(j % 2000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        print("loss",sum((c-d)**2))
        print("------------")
        print("            ")
