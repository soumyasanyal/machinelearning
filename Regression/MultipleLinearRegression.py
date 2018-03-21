
# coding: utf-8

# # Introduction

# This approach to model multiple linear regression (MLR) is based on using vectored approach to solve for minimizing the residual sum of sqaures. Though we don't actually solve the RSS here, I'll try to find some understanding for the solution ( needs some research ). In MLR, the 'linear' means that the hypothesis function is linear in coefficients. It can be written as below,
# $$y = \alpha + \beta_1x_1 + \beta_2x_2 + ... \beta_px_p$$
# where $\alpha$ is the intercept and $\beta_j$'s are the coefficients.
# 
# Writing the same in vectorized form,
# $$Y = X * \beta$$
# where $Y$ is the response vector, $\beta$ is the coefficient vector (first coefficient is $\alpha$) and $X$ is the feature matrix (prepended with a vector of 1's to take $\alpha$ into account).
# 
# The RSS equation that we want to minimize is as follows,
# $$RSS = \sum_{i}(y_i - \alpha - \beta_1x_{i,1} - ... - \beta_px_{i,p})^2$$
# 
# This is minimized by the following,
# $$\hat{\beta} = (X^TX)^{-1}X^TY$$
# 
# The evaluation of the accuracy of the algorithm is done using the R-squared metric. This is better that RMSE as the value is agnostic of the dimension of $Y$. The R-squared metric is defined as
# $$R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$
# where $RSS$ is the residual sum of squares and $TSS$ is the total sum of squares.

# # Multiple Linear Regression in python

# We define the vectors and matrices as Numpy arrays. Specifically, Y is the response vector, B is the coefficient vector and X is the feature matrix.

# In[58]:

import numpy as np
from numpy.linalg import inv

class MultipleLinearRegression:
    
    def __init__(self, n):
        #Here n is the number of indicator variables (feature variables).
        #Actually, n+1 coefficients need to be estimated (including the intercept alpha)
        self.B = np.zeros(n+1)

    def __mean(self, X):
        return sum(X) / float(len(X))
    
    def __squareSum(self, X, Y):
        return sum((X - Y)**2)
    
    def fit(self, X, Y):
        X_transpose = X.transpose()
        X_product = np.matmul(X_transpose, X)
        self.B = np.matmul(np.matmul(inv(X_product), X_transpose), Y)
        return self.B
    
    def predict(self, X_test):
        return np.matmul(X_test, self.B)
    
    def score(self, Y_test, Y_predict):
        return 1 - self.__squareSum(Y_test, Y_predict) / self.__squareSum(Y_test, self.__mean(Y_test))


# In[67]:

import random

def randomSampleGenerator(n):
    Y = []
    X = []
    random.seed(10)
    for i in range(1,n+1):
        x1 = random.uniform(-50, 50)
        x2 = random.uniform(-10, 10)
        var = random.random() * 50 * random.randint(-1, 1)
        y = (3 + 4 * x1 + 7 * x2) + var
        Y.append(y)
        X.append([1, x1, x2])
    return [np.array(X), np.array(Y)]

if __name__ == "__main__":
    N_samples = 1000
    [X, Y] = randomSampleGenerator(N_samples)
    print('Sample function which is being predicted: y = 3 +4*x1 + 7*x2')
    #print('Sample feature matrix:\n', X)
    #print('Response vector:', Y)
    [X_train, Y_train ] = [X[0:int(0.7*N_samples),:], Y[0:int(0.7*N_samples)]]
    [X_test, Y_test ] = [X[int(0.7*N_samples):N_samples,:], Y[int(0.7*N_samples):N_samples]]
    #print('Training set:\n', X_train)
    #print(Y_train)
    #print('Test set:\n', X_test)
    #print(Y_test)
    MLR = MultipleLinearRegression(2)
    coeff = MLR.fit(X_train, Y_train)
    print('Optimized Coefficients:', coeff)
    Y_predict = MLR.predict(X_test)
    #print('Prediction YPredict: ', Y_predict)
    print('R squared score: ', MLR.score(Y_test, Y_predict))
