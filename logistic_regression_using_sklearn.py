#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:17:11 2024

@author: meghanapuli
"""

import numpy as np

# Dataset
X= np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

from sklearn.linear_model import LogisticRegression

# Fit the model
lr_model = LogisticRegression()
lr_model.fit(X, y)

# Make predictions
y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)

# Calculate accuracy
print("Accuracy on training set:", lr_model.score(X, y))

X_test = [[]]
X_test[0].append(float(input("\nEnter x0 value: ")))
X_test[0].append(float(input("Enter x1 value: ")))

prediction = lr_model.predict(X_test)

print("Model's prediction: ", prediction[0])