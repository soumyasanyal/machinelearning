
# coding: utf-8

# # Introduction

# Decision trees is one of the basic supervised learning algorithm which can be used for both regression and classification problems. We focus specifically on the classification problem only. In this problem, the aim is to build a tree starting from root at the top to the leaves in the bottom with specific decision as each nodes (questions that can separate the dataset into two halves). The whole algorithm can be divided into some steps:
#     1. Generating list of valid questions
#     2. Using some metric to measure the effectiveness of each question
#     3. Choosing the optimal question and splitting the dataset
#     4. Iterating on the subsets, stopping once no split is possible (homogeneous class) or some depth criteria
# This is a very simple approach to learn a decision tree model. To measure the effectiveness of a question we use the concepts of Gini Impurity and Information Gain.

# #### Gini Impuity

# This measures the chance of a randomly picked sample being incorrectly labelled if the label is randomly picked from the given label distribution in the set. This can be written as:
# $$I_G(p) = \sum_{i=1}^{J}p_i(1 - p_i) = 1 - \sum_{i=1}^{J}p_i^2$$
# where $p_i$ is the probability of $i^{th}$ label and $J$ is the class size.

# #### Information Gain

# This is the change in the entropy from previous state to the current state. More formally,
# $$IG(T, a) = H(T) - H(T|a)$$
# where $H(T) = -\sum p_i log_2(p_i)$ is the entropy of the system in original state and $H(T|a)$ is the weighted average entropy of the system after action $a$.
# This can be defined using Gini Impurity as well. Given we have sample $S$ and action/question $a$ generates $V$ subsets $S_i$. Then, information gain is given by:
# $$IG(S, a) = I_G(S) - \sum_{i=1}^{V}\frac{|S_i|}{|S|}I_G(S_i)$$

# #### Generating questions and splits

# To generate valid questions we need to iterate over all features to get the unique values, setup equality/inequality questions about them. For numerical values we set greater than equal to condition and for categorical values we set equality condition as question criteria. Based on this, the dataset is split into true set and false set and Information Gain is estimated based on Gini Impurity.

# # Decision trees in python

# We need to define valid questions given a dataset, functions to calculate gini impurity, information gain for given splits and then decide on the optimal split. Then a function can use all these facilities to build a decision tree from a given decision tree. The dataset is defined as feature matrix $X$ and the class labels/target vector $Y$. For defining the questions, we need the specific feature and the numeric/text value i.e. something like (age, 5) which means that the question formed is 'Is age greater than or equal to 5'. Similarly, for text based questions it could mean that (gender, female) means the question 'Is the gender equal to female'. The dataset split happens such that the true cases go to left and the false cases to the right.

# In[42]:

import numpy as np
from collections import Counter

class TreeNode:
    
    def __init__(self, nodeType='decision', attribute=None, left=None, right=None):
        self.nodeType = nodeType    # Node can be of type 'decision' or 'leaf'
        self.attribute = attribute  # For decision -> question(of type list)
                                    # For leaf -> class(of type string/counter)
        self.left = left            # Reference to left subtree
        self.right = right          # Pointer to right subtree

    def printTree(self):
        print(self.nodeType, self.attribute)
        if self.nodeType == 'leaf':
            return
        if self.left is None:
            print('No leaf in left')
        else:
            self.left.printTree()
        if self.right is None:
            print('No leaf in right')
        else:
            self.right.printTree()

class DecisionTree:
    
    def __init__(self, metric='gini', minLeaves=2):
        self.metric = metric         # the metric used for quantifying questions (only gini supported for now)
        self.minLeaves = minLeaves   # the minimum size of dataset required for splitting a node
        self.rootNode = None         # Internal pointer to the root node of the decision tree learned
    
    def getRootNode(self):
        return self.rootNode
    
    def __countClassLabel(self, Y):
        return Counter(Y)
    
    def __isNumeric(self, val):
        return isinstance(val, int) or isinstance(val, float)
    
    def __giniImpurity(self, Y):
        LabelCounts = self.__countClassLabel(Y)
        giniImpurity = 1
        for label in LabelCounts:
            giniImpurity -= (LabelCounts[label] / float(len(Y))) ** 2
        return giniImpurity
    
    def __informationGain(self, leftSplitY, rightSplitY, parentGiniImpurity):
        weightLeft = len(leftSplitY) / float(len(leftSplitY) + len(rightSplitY))
        return parentGiniImpurity - weightLeft * self.__giniImpurity(leftSplitY)                     - (1 - weightLeft) * self.__giniImpurity(rightSplitY)
    
    def __uniqueLabels(self, X, featureIndex):
        return set([x[featureIndex] for x in X])   #select all rows for given feature index
    
    def __createQuestions(self, X):
        # generate the possbile valid questions in the format:
        # [feature index, value] i.e. [0, 5] => 'age greater than equal to 5' if
        # 0th feature is age.
        validQuestions = []
        for index in range(len(X[0])):
            uniqueVals = self.__uniqueLabels(X, index)
            validQuestions.extend([index, val] for val in uniqueVals)
        return validQuestions
    
    def __match(self, x, question):
        # x is a row from feature matrix X
        [featureIndex, value] = question
        if self.__isNumeric(value):
            return x[featureIndex] >= value
        else:
            return x[featureIndex] == value

    def __splitDataset(self, X, Y, question):
        # question is an item from the list of valid questions of the format [feature index, value]
        leftX = []; rightX = []; leftY = []; rightY = []
        for idx, row in enumerate(X):
            if self.__match(row, question):
                leftX.append(row)
                leftY.append(Y[idx])
            else:
                rightX.append(row)
                rightY.append(Y[idx])
        return [leftX, leftY, rightX, rightY]
    
    def __bestQuestion(self, X, Y):
        # find the optimal question for a given dataset. Also, return the corresponding gain
        bestGain = 0
        bestQuestion = []
        allQuestions = self.__createQuestions(X)
        parentGini = self.__giniImpurity(Y)
        for question in allQuestions:
            [leftX, leftY, rightX, rightY] = self.__splitDataset(X, Y, question)
            gain = self.__informationGain(leftY, rightY, parentGini)
            if gain > bestGain:
                bestGain = gain
                bestQuestion = question
        return [bestGain, bestQuestion]
    
    def __fitHelper(self, X, Y):
        [gain, question] = self.__bestQuestion(X, Y)
        
        # if no gain or minLeaves reached, this means this is a leaf node. For now, class is predicted
        # as the majority in the split.
        if gain == 0 or len(Y) <= self.minLeaves:
            # classValue = self.__countClassLabel(Y).most_common(1)[0][0]
            classValue = self.__countClassLabel(Y)
            return TreeNode(nodeType='leaf', attribute=classValue)
        
        [leftX, leftY, rightX, rightY] = self.__splitDataset(X, Y, question)
        leftNode = self.__fitHelper(leftX, leftY)
        rightNode = self.__fitHelper(rightX, rightY)
        
        return TreeNode(nodeType='decision', attribute=question, left=leftNode, right=rightNode)
    
    def fit(self, X, Y):
        self.rootNode = self.__fitHelper(X, Y)
    
    def printDecisionTree(self):
        if self.rootNode is None:
            print('Cannot print tree with none root type!!')
        self.rootNode.printTree()
    
    def __predictHelper(self, x, currentNode):
        if currentNode.nodeType == 'leaf':
            # At this point, for now, just return the most obvious choice for class
            return currentNode.attribute.most_common(1)[0][0]
        
        if self.__match(x, currentNode.attribute):
            return self.__predictHelper(x, currentNode.left)
        else:
            return self.__predictHelper(x, currentNode.right)
    
    def predict(self, X):
        return [self.__predictHelper(x, self.rootNode) for x in X]


# In[44]:

X = [['Green', 3],
    ['Yellow', 3],
    ['Red', 1],
    ['Red', 1],
    ['Yellow', 3],
    ['Yellow', 4],
    ['Yellow', 5]]
#NOTE: If using numpy array -> dtype object is required to maintain the string/integer types as is.
#If not mentioned, will convert int to string

Y = ['Apple', 'Apple', 'Grape', 'Grape', 'Lemon', 'Apple', 'Apple']

D = DecisionTree()
D.fit(X, Y)
print('Preorder traversal of the decision tree: format<nodetype attribute>')
D.printDecisionTree()
print('\nPredicting on the same set:')
print(D.predict(X))

