#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:50:54 2024

@author: meghanapuli
"""

import numpy as np
import matplotlib.pyplot as plt

dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Function to plot the data
def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors=dlblue, lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False
 
# Plot the given data
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
print("\nGiven data:")
plt.show()

# Compute the sigmoid of z
def sigmoid(z):
      
    g = 1.0/(1.0+np.exp(-z))

    return g

# compute the prediction of the model
def compute_model_output(X, w, b): 
    z = np.dot(w,X) + b
    f_wb = sigmoid(z)
    return f_wb

# Compute the cost of the model
def compute_cost_logistic(X, y, w, b):
    m,n = X.shape
    cost = 0.0
    
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w,X[i]) + b)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    
    cost = cost/m
    return cost

def compute_gradient_logistic(X, y, w, b): 
    m,n = X.shape
    dj_dw = np.zeros((n,))                          
    dj_db = 0.
    
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w,X[i]) + b)
        error = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += error*X[i,j]
        
        dj_db += error
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    return dj_db, dj_dw

# Gradient descent to find optimal w,b
def gradient_descent(X, y, w_in, b_in, gradient_function, alpha, num_iters): 
    w = w_in
    b = b_in
    
    print("\nIterations vs cost\n")
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i % 1000 == 0 or i == num_iters-1:
            print(f"Iteration {i:4d}:", "Cost:", compute_cost_logistic(X, y, w, b))
    
    return w,b

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alpha = 0.1
iters = 10000

w_out, b_out = gradient_descent(X_train, y_train, w_tmp, b_tmp, compute_gradient_logistic, alpha, iters) 
print(f"\nOptimal parameters: w:{w_out}, b:{b_out}")

fig,ax = plt.subplots(1,1,figsize=(5,4))

# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)

# Plot the decision boundary
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()

X_test = []
X_test.append(float(input("\nEnter x0 value: ")))
X_test.append(float(input("Enter x1 value: ")))

computed_value = compute_model_output(X_test, w_out, b_out)

if computed_value < 0.5:
    output = 0
else:
    output = 1
    
print("Model's output: ", output)