'''
Decision Trees

@author: Abraham Zakharov
'''
import numpy as np
import math

def entropy (numOne, numTwo):
    if numOne == 0 or numTwo == 0:
        return 0
    entropy = (-1 * (numOne/(numOne + numTwo)) * math.log(numOne/(numOne + numTwo), 2)) - ((numTwo/(numOne + numTwo)) * (math.log(numTwo/(numOne + numTwo),2)))
    return entropy
    
def IG(D, index, value):
    """Compute the Information Gain of a split on attribute index at value
    for dataset D.
    
    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Information Gain for the given split
    """
    totalYes = 0
    totalNo = 0
    for points in D[1]:
        if points==1:
            totalYes = totalYes+1
        else:
            totalNo = totalNo + 1
 
    entropySystem = (entropy(totalYes,totalNo))
    
    less = 0 
    greater = 0
    lessMatch = 0
    lessNonMatch = 0
    greaterMatch = 0
    greaterNonMatch = 0
    
    for index, points in enumerate(D[0].T[index]):
        if (points > value):
            greater = greater + 1
            if D[1][index] == 1:
                greaterMatch = greaterMatch + 1
            else:
                greaterNonMatch = greaterNonMatch + 1
        else:
            less = less + 1 
            if D[1][index] == 1:
                lessMatch = lessMatch + 1
            else:
                lessNonMatch = lessNonMatch + 1
                
    eCheck = (less/(less+greater)) * entropy(lessMatch, lessNonMatch) + (greater/(less+greater)) * entropy(greaterMatch, greaterNonMatch)
    
    IGain = entropySystem - eCheck
    return IGain

def G(D, index, value):
    """Compute the Gini index of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Gini index for the given split
    """
    less = 0 
    greater = 0
    lessMatch = 0
    lessNonMatch = 0
    greaterMatch = 0
    greaterNonMatch = 0
    
    for index, points in enumerate(D[0].T[index]):
        if (points > value):
            greater = greater + 1
            if D[1][index] == 1:
                greaterMatch = greaterMatch + 1
            else:
                greaterNonMatch = greaterNonMatch + 1
        else:
            less = less + 1 
            if D[1][index] == 1:
                lessMatch = lessMatch + 1
            else:
                lessNonMatch = lessNonMatch + 1
    
    
    if (greater == 0):
        GiniGreaterMatch = 0
    else:
        GiniGreaterMatch = (1 - ((greaterMatch/greater)*(greaterMatch/greater))) - ((greaterNonMatch/greater)*(greaterNonMatch/greater))
        
    GiniLessMatch = (1 - ((lessMatch/less)*(lessMatch/less))) - ((lessNonMatch/less)*(lessNonMatch/less))
    Gini = ((greater/(greater+less)) * GiniGreaterMatch) + ((less/(greater+less)) * GiniLessMatch) 
    return Gini

def CART(D, index, value):
    """Compute the CART measure of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the CART measure for the given split
    """
    less = 0 
    greater = 0
    lessMatch = 0
    lessNonMatch = 0
    greaterMatch = 0
    greaterNonMatch = 0
    
    
    for index, points in enumerate(D[0].T[index]):
        if (points > value):
            greater = greater + 1
            if D[1][index] == 1:
                greaterMatch = greaterMatch + 1
            else:
                greaterNonMatch = greaterNonMatch + 1
        else:
            less = less + 1 
            if D[1][index] == 1:
                lessMatch = lessMatch + 1
            else:
                lessNonMatch = lessNonMatch + 1
    
    #print(less, lessMatch, lessNonMatch, greater, greaterMatch, greaterNonMatch)
        
    if (greater == 0):
        cartVal = 2 *  abs((lessMatch/less) - (lessNonMatch/less))
    else:
        cartVal = 2 * (greater/(greater+less)) * (less/(greater+less)) * (abs((greaterMatch/greater)-(greaterNonMatch/greater)) + abs((lessMatch/less) - (lessNonMatch/less)))
    
    return cartVal
 
def bestSplit(D, criterion):
    """Computes the best split for dataset D using the specified criterion

    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes
        criterion: one of "IG", "GINI", "CART"

    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """

    bestIndex = 0
    bestPoint = 0
    bestCalc = 0
    #functions are first class objects in python, so let's refer to our desired criterion by a single name
    if criterion == "IG":
        for i in range(len(D[0])):
            for j in range(len(D[0][i])):
                IGain = IG(D, j, D[0][i][j])
                if (IGain > bestCalc):
                    #print(IGain)
                    bestCalc = IGain
                    bestIndex = j
                    bestPoint = D[0][i][j]
        return (bestIndex, bestPoint)
    
    if criterion == "GINI":
        bestCalc = 1
        for i in range(len(D[0])):
            for j in range(len(D[0][i])):
                GINI = G(D, j, D[0][i][j])
                if (GINI < bestCalc):
                    bestCalc = GINI
                    bestIndex = j
                    bestPoint = D[0][i][j]
        return (bestIndex, bestPoint)       
    
    if criterion == "CART":
        for i in range(len(D[0])):
            for j in range(len(D[0][i])):
                CartSum = CART(D, j, D[0][i][j])
                if (CartSum > bestCalc):
                    bestCalc = CartSum
                    bestIndex = j
                    bestPoint = D[0][i][j]
        return (bestIndex, bestPoint)

def load(filename):
    """Loads filename as a dataset. Assumes the last column is classes, and 
    observations are organized as rows.

    Args:
        filename: file to read

    Returns:
        A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
        where X[i] comes from the i-th row in filename; y is a list or ndarray of 
        the classes of the observations, in the same order
    """
    X = np.genfromtxt(filename, delimiter=",", usecols=range(0,10))
    Y = np.genfromtxt(filename, delimiter=",", usecols=(10))
    return (X, Y)
    

def classifyIG(train, test):
    """Builds a single-split decision tree using the Information Gain criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    split = bestSplit(train, "IG")
    index = split[0]
    value = split[1]
    testList = []
    for i in range(len(test[0].T[index])):
        if (test[0].T[index][i] > value):
            testList.append(1)
        else:
            testList.append(0)
    return testList
    
def classifyG(train, test):
    """Builds a single-split decision tree using the GINI criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    split = bestSplit(train, "GINI")
    index = split[0]
    value = split[1]
    testList = []
    for i in range(len(test[0].T[index])):
        if (test[0].T[index][i] > value):
            testList.append(1)
        else:
            testList.append(0)
    return testList

def classifyCART(train, test):
    """Builds a single-split decision tree using the CART criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    split = bestSplit(train, "CART")
    index = split[0]
    value = split[1]
    testList = []
    for i in range(len(test[0].T[index])):
        if (test[0].T[index][i] > value):
            testList.append(1)
        else:
            testList.append(0)
    return testList

def main():
    """This portion of the program will run when run only when main() is called.
    This is good practice in python, which doesn't have a general entry point 
    unlike C, Java, etc. 
    This way, when you <import HW2>, no code is run - only the functions you
    explicitly call.
    """
    #part (e) - import train.txt and find best splits for each criteria
    train = load('train.txt')
    print(train)
    print ("Best Split by Information Gain: ")
    print (bestSplit(train, "IG"))
    print("Best Split by GINI: ") 
    print (bestSplit(train, "GINI"))
    print ("Best Split by CART: ") 
    print (bestSplit(train, "CART"))
    test = load('test.txt')
    print(test)
    print ("Predicted Class by Information Gain: ")
    print(classifyIG(train, test))
    print ("Predicted Class by GINI: ")
    print(classifyG(train, test))
    print ("Predicted Class by CART: ")
    print(classifyCART(train, test))

if __name__=="__main__": 
    """__name__=="__main__" when the python script is run directly, not when it 
    is imported. When this program is run from the command line (or an IDE), the 
    following will happen; if you <import HW2>, nothing happens unless you call
    a function.
    """
    main()
    pass
