'''
    Author:
    Arpit Parwal <aparwal@usc.edu>
    Yeon-soo Park <yeonsoop@usc.edu>
'''

# Imports
import numpy as np
import math
# End imports


def printResult(nIterations,model,accuracy,correctPrediction,TotalData):
    print('Total Iterations done: {0}'.format(nIterations))
    print('Final Weights [W0, W1, W2,...]: {0}'.format(model.weights))
    print('\n\t\t'.join(['Accuracy{0}','Predicted: {1}','Total Samples: {2}']).format(accuracy, correctPrediction, TotalData))


class LogisticRegression():
    '''
        Implements Logistic Regression.
    '''
    # finding gradient using sigmoid
    @staticmethod
    def findGradient_sigmoid(weights, x, y):
        # using the sigmoid formula n Ɵ(s) = es/(1+es).
        arg = y * np.dot(weights, x)
        tmp = 1 + np.exp(arg)
        res = (x * y) / tmp
        return res

    # functn to find out the prdiction probability
    @staticmethod
    def findProbability(weights, x, y):
        # using the sigmoid formula n Ɵ(s) = es/(1+es).
        arg = y * np.dot(weights, x)
        tmp = np.exp(arg)
        return tmp / (1 + tmp)

    #  initiaise fnct to set weights, learning rate (learningRate) and max iteration
    #  default values are used in case there is no value given
    def __init__(self, weights=[], learningRate=0.001, maxIter=1000):
        self.weights = weights
        self.learningRate = learningRate
        self.maxIter = maxIter

    # training the model using this function

    def train(self, X, Y, verbose=False):
        # getting the numebr of elements in the array and it's dimension
        N, d = X.shape
        # setting a default of 1 value for the coordiantes
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.random.random(d+1)
        iter = 0
        while iter < self.maxIter:
            # introductin gradient to be initialised as a zero matrix with dimension as d+1 of the dataset dimentsion
            gradient = np.zeros(d+1)
            # using zip fnctn to get pairwise iterations of label and coordinates
            for x, y in zip(X, Y):
                # iteratively calling gradient function and altering the gradient curve
                gradient = np.add(
                    gradient, LogisticRegression.findGradient_sigmoid(self.weights, x, y))
            # normalisation
            gradient /= N
            # adding the weights along with learning factor
            self.weights += self.learningRate * gradient
            iter += 1

            # just fr checking progress
            if verbose:
                if iter % 500 == 0:
                    print('Completed iterations: {0}'.format(iter))

        return iter

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        # intialisin ths array to store result
        Y = [None for _ in range(X.shape[0])]
        # finding out the probablility of prediction, in case we found more than half porbabiity we will assume it to be true
        # however, this can be normalised as there would be a lot of predictions around 0.5 which should be also taken as 1
        for idx, x in enumerate(X):
            prob_1 = LogisticRegression.findProbability(self.weights, x, 1)
            if prob_1 > 0.5:
                Y[idx] = 1
            else:
                Y[idx] = -1

        return np.asarray(Y)



if __name__ == '__main__':

    # defining the learningRate value
    learningRate = 0.05
    #  definin the max iterations
    maxIter = 7000
    # defining the source file for the data
    srcFilePath = 'classification.txt'
    # getting the data in format to be stored inside the data np array
    # using only the 1st 2nd 3rd and 5th column here
    dataArray = np.loadtxt(srcFilePath,
                           delimiter=',',
                           dtype='float',
                           usecols=(0, 1, 2, 4)
                           )

    # Seperting the points and keeping it in X array
    X = dataArray[:, :-1]
    #  Seperating the classification lable and putting it in Y array
    Y = dataArray[:, -1]

    # calling the logisitic regression to train the model
    model = LogisticRegression(learningRate=learningRate, maxIter=maxIter)

    # calling this function to train the model we just defined.
    nIterations = model.train(X, Y, verbose=True)

    # once our model is trained we call the prediction function on the same dataset
    # this should not be the case, we need to have a separate test dataset, but anyhow
    YPred = model.predict(X)
    # finding correct interpretations
    correctPrediction = np.where(Y == YPred)[0].shape[0]
    TotalData = YPred.shape[0]
    accuracy = correctPrediction / TotalData
    # utility to print data
    printResult(nIterations,model,accuracy,correctPrediction,TotalData)
   
 
