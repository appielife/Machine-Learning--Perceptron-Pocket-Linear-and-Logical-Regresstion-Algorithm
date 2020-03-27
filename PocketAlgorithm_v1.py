'''
    Author:
    Arpit Parwal <aparwal@usc.edu>
    Yeon-soo Park <yeonsoop@usc.edu>
'''
import numpy as np
import matplotlib.pyplot as plt
import copy

class PocketAlgorithm:

    def __init__(self, datapoints, no_of_inputs, iteration=7000, learning_rate=0.0001):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, no_of_inputs + 1)
        self.datapoints = datapoints
        self.plotResult = list()
        self.bestResult = float("inf")
        self.bestWeights = np.array

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = -1
        return activation

    def train(self):
        training_inputs, labels = self.datapoints[:,:-2], self.datapoints[:,-1:]
        misclassified = 1
        iteration = 0

        while iteration < self.iteration:
            misclassified = 0
            iteration += 1

            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error_rate = 1 if label == 1 else -1

                if (label == 1 and prediction == -1) or (label == -1 and prediction == 1):
                    misclassified += 1
                    self.weights[1:] += self.learning_rate * error_rate * inputs
                    self.weights[0] += self.learning_rate * error_rate

            self.plotResult.append(misclassified)

            if misclassified < self.bestResult:
                self.bestResult = misclassified
                self.bestWeights = copy.deepcopy(self.weights)

            if iteration % 500 == 0:
                print("Iteration {}, misclassified points {}, Evaluation {}%".format(iteration, misclassified, self.evaluate()))

        print("")
        print("======== Result ========= ")
        print("Iteration {}, misclassified points {}".format(iteration, misclassified))
        print("Evaluation {}%".format(self.evaluate()))

    def evaluate(self):
        correct = 0
        training_inputs, labels = self.datapoints[:,:-2], self.datapoints[:,-1:]
        for inputs, label in zip(training_inputs, labels):
            prediction = self.predict(inputs)
            if (label == 1 and prediction == 1) or (label == -1 and prediction == -1):
                correct += 1

        _acc = correct / float(len(training_inputs)) * 100.0
        return _acc

    def plot(self):

        total_data = len(self.datapoints)

        with np.printoptions(precision=7, suppress=True):
            print("Minimum Misclassified Points/Best Result: ", self.bestResult)
            print("Weight After Final iteration: ", self.weights.transpose())
            print("Best Weights of Pocket: ", self.bestWeights.transpose())
            print("Best Accuracy of Pocket: {}%".format(((total_data - self.bestResult) / float(total_data)) * 100))

        plt.plot(np.arange(0, self.iteration), self.plotResult)
        plt.xlabel("Iterations")
        plt.ylabel("Misclassified Points")
        plt.axis([0, self.iteration, 800, 1200])
        plt.show()

def getInputData(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data


if __name__ == '__main__':

    data_points = np.array(getInputData('classification.txt'))
    no_of_inputs = 3
    pck = PocketAlgorithm(data_points, no_of_inputs)
    pck.train()
    pck.evaluate()
    pck.plot()
