from random import random
import math
from matplotlib import pyplot

'''
HELPER FUNCTIONS
'''
def target(row, theta):
    return ((float(row[0]) * theta[0]) + (float(row[1]) * theta[1]) + (float(row[2]) * theta[2]) + (float(row[3]) * theta[3]) + bias[0])

def sigmoid(target):
    return 1 / (1 + math.exp(-1 * float(target)))

def delta_theta(target, category, x):
    return 2 * (sigmoid(target) - float(category)) * (1 - sigmoid(target)) * sigmoid(target) * float(x)

def delta_bias(target, category):
    return 2 * (sigmoid(target) - float(category)) * (1 - sigmoid(target)) * sigmoid(target)

def prediction(sigmoid):
    return round(sigmoid)

def error(prediction, actual):
    return (prediction - actual) ** 2

data = open('iris_data.csv')

categories = {
    'Iris-setosa': '00',
    'Iris-versicolor': '01',
    'Iris-virginica': '10',
}

thetas = [([random()] * 4), ([random()] * 4)]

bias = [random()] * 2

learning_rate = 0.1

epoch = 100

'''
DATA PRE-PROCESSING
'''
data = data.read().split('\n')

# Re-arrange data to even up the data.
for i in range(0, 50): 
    if (i + 1) % 3 == 2:
        data[i], data[i + 50] = data[i + 50], data[i]
    elif (i + 1) % 3 == 0:
        data[i], data[i + 100] = data[i + 100], data[i]

'''
MODEL TRAINING & TESTING
'''
accuracies_train = [0] * epoch
errors_train = [0] * epoch
accuracies_test = [0] * epoch
errors_test = [0] * epoch

for iteration in range(0, epoch):
    for k in range(1, 6):
        test = data[((k-1) * 30):(k * 30)]
        train = data[0:((k-1) * 30)] + data[(k * 30):]

        '''
        MODEL TRAINING
        '''
        for row in train:
            row = row.split(',')

            targets = [target(row, thetas[0]), target(row, thetas[1])]
            sigmoids = [sigmoid(targets[0]), sigmoid(targets[1])]
            predictions = [prediction(sigmoids[0]), prediction(sigmoids[1])]

            for i in range(0, 2):
                category = float(categories[row[4]][i])
                correct = (predictions[i] == category)

                errors_train[iteration] += error(sigmoids[i], category)
                accuracies_train[iteration] += (1 if correct else 0)

                for j in range(0, 4):
                    thetas[i][j] -= delta_theta(targets[i], category, row[j])
                bias[i] -= delta_bias(targets[i], category)

        errors_train[iteration] /= len(train)
        accuracies_train[iteration] /= len(train)

        '''
        MODEL TESTING
        '''
        for row in test:
            row = row.split(',')

            targets = [target(row, thetas[0]), target(row, thetas[1])]
            sigmoids = [sigmoid(targets[0]), sigmoid(targets[1])]
            predictions = [prediction(sigmoids[0]), prediction(sigmoids[1])]
            category = [float(categories[row[4]][0]), float(categories[row[4]][1])]

            correct = (predictions[0] == category[0])
            errors_test[iteration] += error(sigmoids[0], category[0])
            accuracies_test[iteration] += (1 if correct else 0)

            correct = (predictions[1] == category[1])
            errors_test[iteration] += error(sigmoids[1], category[1])
            accuracies_test[iteration] += (1 if correct else 0)

        errors_test[iteration] /= len(test)
        accuracies_test[iteration] /= (len(test) * 2)

pyplot.plot(range(1, epoch+1), errors_train, label='Train')
pyplot.plot(range(1, epoch+1), errors_test, label='Test')
pyplot.legend()
pyplot.ylabel('Error')
pyplot.xlabel('Epoch')
pyplot.show()

pyplot.plot(range(1, epoch+1), accuracies_train, label='Train')
pyplot.plot(range(1, epoch+1), accuracies_test, label='Test')
pyplot.legend()
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.show()
