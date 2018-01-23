# https://github.com/FonzieTree
import numpy as np
import copy
np.random.seed(0)
 
# define sigmoid and its derivative functions
def sigmoid(x):
    return 1/(1+np.exp(x))
def sigmoid_derivative(x):
    return x*(1-x)
 
# define hello, x is hell, y is ello
'''
chars 'h','e','l','o',is represented as follows
'h' <---[1,0,0,0]
'e' <---[0,1,0,0]
'l' <---[0,0,1,0]
'o' <---[0,0,0,1]
'''
 
# some parameters
epoch = 20000
lr = 0.1
seq_len = 4
alpha = 0.1
mc = 1
x = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]])
y = np.array([[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]])
w1 = alpha * np.random.randn(4,4)
wh = alpha * np.random.randn(4,4)
w2 = alpha * np.random.randn(4,4)
h0 = np.zeros((1,4))
for i in range(epoch):
    # compute hidden layers
    q1 = np.dot(x[0],w1)
    p1 = np.dot(h0,wh)
    h1 = p1 + sigmoid(q1)
    q2 = np.dot(x[1],w1)
    p2 = np.dot(h1,wh)
    h2 = p2 + sigmoid(q2)
    q3 = np.dot(x[2],w1)
    p3 = np.dot(h2,wh)
    h3 = p3 + sigmoid(q3)
    q4 = np.dot(x[3],w1)
    p4 = np.dot(h3,wh)
    h4 = p4 + sigmoid(q4)
 
    # compute output
    o1 = np.dot(h1,w2)
    yhat1 = np.exp(o1)
    yhat1 = yhat1/np.sum(yhat1)
    o2 = np.dot(h2,w2)
    yhat2 = np.exp(o2)
    yhat2 = yhat2/np.sum(yhat2)
    o3 = np.dot(h3,w2)
    yhat3 = np.exp(o3)
    yhat3 = yhat3/np.sum(yhat3)
    o4 = np.dot(h4,w2)
    yhat4 = np.exp(o4)
    yhat4 = yhat4/np.sum(yhat4)
 
    # do backpropagation
    loss1 = -np.log(yhat1[0][np.argmax(y[0])])
    loss2 = -np.log(yhat2[0][np.argmax(y[1])])
    loss3 = -np.log(yhat3[0][np.argmax(y[2])])
    loss4 = -np.log(yhat4[0][np.argmax(y[3])])
    loss = (loss1 + loss2 + loss3 + loss4)/seq_len
    dscore1 = copy.deepcopy(yhat1)
    dscore2 = copy.deepcopy(yhat2)
    dscore3 = copy.deepcopy(yhat3)
    dscore4 = copy.deepcopy(yhat4)
    dscore1[0][np.argmax(y[0])] = dscore1[0][np.argmax(y[0])] - 1
    dscore2[0][np.argmax(y[1])] = dscore2[0][np.argmax(y[1])] - 1
    dscore3[0][np.argmax(y[2])] = dscore3[0][np.argmax(y[2])] - 1
    dscore4[0][np.argmax(y[3])] = dscore4[0][np.argmax(y[3])] - 1
 
    # path4
    do4 = sigmoid_derivative(dscore4)
    dh4 = np.dot(do4,w2.T)
    dw24 = np.dot(h4.T,o4)
    dp4 = dh4
    dq4 = sigmoid_derivative(dh4)
    dw14 = np.dot(x[3].reshape(seq_len,1),dq4)
 
 
    # path3
    do3 = sigmoid_derivative(dscore3)
    dh3 = np.dot(do3,w2.T) + np.dot(dp4,wh.T)
    dw23 = np.dot(h3.T,o3)
    dp3 = dh3
    dq3 = sigmoid_derivative(dh3)
    dw13 = np.dot(x[2].reshape(seq_len,1),dq3)
    dwh3 = np.dot(h3.T,dp4)
 
    # path2
    do2 = sigmoid_derivative(dscore2)
    dh2 = np.dot(do2,w2.T) + np.dot(dp3,wh.T)
    dw22 = np.dot(h2.T,o2)
    dp2 = dh2
    dq2 = sigmoid_derivative(dh2)
    dw12 = np.dot(x[1].reshape(seq_len,1),dq2)
    dwh2 = np.dot(h2.T,dp3)
 
    # path1
    do1 = sigmoid_derivative(dscore1)
    dh1 = np.dot(do1,w2.T) + np.dot(dp2,wh.T)
    dw21 = np.dot(h1.T,o1)
    dp1 = dh1
    dq1 = sigmoid_derivative(dh1)
    dw11 = np.dot(x[0].reshape(seq_len,1),dq1)
    dwh1 = np.dot(h1.T,dp2)
     
    # path0
    dh0 = np.dot(dp1,wh.T)
    dwh0 = np.dot(h0.T,dp1)
 
     
    # summarize
    dw1 = dw14 + dw13 + dw12 + dw11
    dwh = dwh3 + dwh2 + dwh1 + dwh0
    dw2 = dw24 + dw23 + dw22 + dw21
 
    # reset w1,wh,w2
    w1 = mc * w1 - lr * dw1
    wh = mc * wh - lr * dwh
    w2 = mc * w2 - lr * dw2
    print(loss)
s = [np.argmax(yhat1), np.argmax(yhat2), np.argmax(yhat3), np.argmax(yhat4)]
print('------------------------')
for i in s:
    if i == 0:
        print('h')
    elif i == 1:
        print('e')
    elif i == 2:
        print('l')
    else:
        print('o')
