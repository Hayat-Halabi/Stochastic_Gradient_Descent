# Stochastic_Gradient_Descent
Applying Stochastic Gradient Descent-26
``` python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split


bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

bc.target_names


bc_x = pd.DataFrame(X, columns=bc.feature_names)
bc_y = pd.DataFrame({'Diagnosis': y})
bc_df = pd.concat([bc_x, bc_y], axis=1) 
bc_df

n_iterations = 100
learning_rate = 0.01
 
def predict(X, y, coef):
    '''
    Activation function: w0 + w1*x1 + w2*x2 + ... + wn*xn
    '''
    output = np.dot(X, coef[1:]) + coef[0]
    '''
    Unit Step function: Predict 1 if output >= 0 else 0
    '''
    return np.where(output >= 0.0, 1, 0)
     
def fit(X, y):
        rgen = np.random.RandomState(1)
        coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(n_iterations):
            for xi, expected_value in zip(X, y):
                predicted_value = predict(xi, target, coef_)
        
                coef_[1:] += learning_rate * (expected_value - predicted_value) * xi
                coef_[0] += learning_rate * (expected_value - predicted_value) * 1
        return coef_

class CustomPerceptron(object):
     
    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
 
    '''
    Stochastic Gradient Descent
     
    1. Weights are updated based on each training examples.
    2. Learning of weights can continue for multiple iterations.
    3. Learning rate needs to be defined.
    '''
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iterations):
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                self.coef_[1:] += self.learning_rate * (expected_value - predicted_value) * xi
                self.coef_[0] += self.learning_rate * (expected_value - predicted_value) * 1
     
    '''
    Activation function calculates the value of weighted sum of input value
    '''
    def activation(self, X):
            return np.dot(X, self.coef_[1:]) + self.coef_[0]
     
    '''
    Prediction is made on the basis of unit step function.
    '''
    def predict(self, X):
        output = self.activation(X)
        return np.where(output >= 0.0, 1, 0)
     
    '''
    Model score is calculated based on comparison of
    expected value and predicted value.
    '''
    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if(target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count)/total_data_count
        return self.score_


prcptrn = CustomPerceptron()

# Training our model 
prcptrn.fit(X_train, y_train)

# Accuracy scores of test data and train data 
prcptrn.score(X_test, y_test), prcptrn.score(X_train, y_train)



#Observation

#The results of the first one are displayed as 0.9064327485380117 for the output for the test and 0.9296482412060302 for the output for the train.


```
Demo downloadable link below 
```

```
