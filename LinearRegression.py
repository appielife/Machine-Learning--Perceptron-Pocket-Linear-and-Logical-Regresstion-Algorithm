'''
    Author:
    Arpit Parwal <aparwal@usc.edu>
    Yeon-soo Park <yeonsoop@usc.edu>
'''
import pandas as pd
import numpy as np
from numpy.linalg import inv

srcFilePath = 'linear-regression.txt'
trainingData=pd.read_csv(srcFilePath,header=None)

# finding inverse (Least Squares soultion)
#  b = (X`X)^-1 X`Y
def runAlgo(X, y):
    return inv(X.T.dot(X)).dot(X.T).dot(y)


#array for storing the classified output 
y=[]
for item in trainingData:
    y=trainingData[2]
    x=list(zip(trainingData[0],trainingData[1]))
X= np.matrix(x)
X = np.column_stack((np.ones(len(X)), X))
#to run linear regressino algorithm
weights=runAlgo(X,y)
# printing the weights returned
print('Final Weights calculated [W0, W1, W2]:=> ___ {0}'.format(weights))
