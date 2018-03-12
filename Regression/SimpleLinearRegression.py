
# coding: utf-8

# # Introduction

# This notebook is my first attempt to write a simple linear regression algorithm from scratch. First we need to define the hypothesis funtion. We know its a straigh line equation for linear regression.

# $$y = b_0 + b_1 * x$$

# Next, we need to find the optimal solution for the above coefficients $b_0$ and $b_1$. For that we can optimize the Residual Sum of Squares given by the expression,$$RSS = \sum_{1}^{n} {e_i}^2$$ where $e_i = y_i - \hat{y_i}$

# This comes out to be (after sufficient mathematics)
# $$b_1 = \frac{\sum_{i=1}^{n}(x_i-\bar{x})*(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2}$$
# $$b_0 = \bar{y} - b_1*\bar{x}$$
# where $\bar{x}$ and $\bar{y}$ are sample means.

# The evaluation of the accuracy of the algorithm is done using the R-squared metric. This is better that RMSE as the value is agnostic of the dimension of $Y$. The R-squared metric is defined as
# $$R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$
# where $RSS$ is the residual sum of squares and $TSS$ is the total sum of squares.

# In this discussion, we'll reprresent the data in the form of vectors $X$ and $Y$. All python code will assume these vectors to be Numpy arrays.

# # Linear Regression in python

# First we need functions to find the mean and variance. Using that we define the covariance for $X$ and $Y$. Next we need to evaluate the coefficients $b_0$ and $b_1$ using the above formula. The prediction function used is the same linear regression form written above. To score the algorithm accuracy we use the R-squared approach.

# In[5]:

import numpy as np

class SimpleLinearRegression:
    
    def __init__(self):
        self.b0 = 0   #intercept
        self.b1 = 0   #slope

    def __mean(self, X):
        return sum(X) / float(len(X))

    def __var(self, X):
        return sum((X - self.__mean(X))**2) / float(len(X) - 1)

    def __cov(self, X, Y):
        return sum((X - self.__mean(X))*(Y - self.__mean(Y))) / float(len(X) - 1)

    #Given vectors X and Y, finds the optimal coefficients b0 and b1
    def fit(self, X, Y):
        self.b1 = self.__cov(X, Y) / self.__var(X)
        self.b0 = self.__mean(Y) - self.b1 * self.__mean(X)

    def predict(self, X):
        return self.b0 + self.b1 * X
    
    def __squareSum(self, X, Y):
        return sum((X - Y)**2)

    def score(self, Y, Y_predict):
        return 1 - self.__squareSum(Y, Y_predict) / self.__squareSum(Y, self.__mean(Y))


# In[6]:

L = SimpleLinearRegression()
X_train = np.array([1, 2, 4, 3, 5])
Y_train = np.array([1, 3, 3, 2, 5])
print('X_train: ', X_train)
print('Y_train: ', Y_train)
print('Fitting Simple Linear Regression model..')
L.fit(X_train, Y_train)
X_test = np.array([1, 2, 4, 3, 5])
Y_test = np.array([1, 3, 3, 2, 5])
print('X_test: ', X_test)
print('Y_test: ', Y_test)
Y_predict = L.predict(X_test)
print('Prediction YPredict: ', Y_predict)
print('R squared score: ', L.score(Y_test, Y_predict))

