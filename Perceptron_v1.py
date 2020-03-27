
'''
    Author:
    Arpit Parwal <aparwal@usc.edu>
    Yeon-soo Park <yeonsoop@usc.edu>
'''
import numpy as np


class Perceptron:

    def __init__(self, datapoints, no_of_inputs, threshold=1000, learning_rate=0.0001, isPocket = False):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, no_of_inputs + 1)
        self.datapoints = datapoints
        self.isPocket = isPocket

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self):
        training_inputs, labels = self.datapoints[:,:-2], self.datapoints[:,-2:-1]
        misclassified = 1
        iteration = 0
        while misclassified != 0:
            misclassified = 0
            iteration += 1
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error_rate = 1 if label == 1 else -1

                if (label == 1 and prediction == 0) or (label == -1 and prediction == 1):
                    misclassified += 1
                    self.weights[1:] += self.learning_rate * error_rate * inputs
                    self.weights[0] += self.learning_rate * error_rate
            if iteration % 50 == 0:
                print("Iteration {}, misclassified points {}, Evaluation {}%".format(iteration, misclassified, self.evaluate()))

        print("")
        print("======== Result ========= ")
        print("Iteration {}, misclassified points {}".format(iteration, misclassified))
        print("Evaluation {}%".format(self.evaluate()))

    def evaluate(self):
        correct = 0
        training_inputs, labels = self.datapoints[:,:-2], self.datapoints[:,-2:-1]
        for inputs, label in zip(training_inputs, labels):
            prediction = self.predict(inputs)
            if (label == 1 and prediction == 1) or (label == -1 and prediction == 0):
                correct += 1

        _acc = correct / float(len(training_inputs)) * 100.0
        return _acc

    def printResult(self):

        print("Weights After Final Iteration: ", np.round(self.weights.transpose(),3))
        #print("Accuracy of Pocket: {}%", self.evaluate())


def getInputData(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data


if __name__ == '__main__':

    data_points = np.array(getInputData('classification.txt'))
    no_of_inputs = 3
    pct = Perceptron(data_points, no_of_inputs)
    pct.train()
    pct.evaluate()
    pct.printResult()