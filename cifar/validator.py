import csv
import numpy as np


def testAccuracy(onTrain, dir):
    if bool(onTrain):
        name = 'training'
    else:
        name = 'testing'
    Labels = np.genfromtxt(dir + name + 'Label.txt', delimiter=' ')
    Datum = np.genfromtxt(dir + name + 'Data.txt', delimiter=' ')
    size = len(Labels)
    accuracy = 0
    for i in range(size):
        indexData = np.argmax(Datum[i])
        indexLabels = np.argmax(Labels[i])
        if indexData == indexLabels:
            accuracy += 1
        else:
            yo = 'yo'
    return(accuracy/size)


dir = 'D:\CIFAR//residual-76%//debugWithOren//New folder//'
dir = 'C:\CIFAR//residual//layer_output//layer_66_dense_1//'
print('train accuracy is ' + str(testAccuracy(True, dir)))
print('test accuracy is ' + str(testAccuracy(False, dir)))
