from __future__ import print_function

import os
import csv
from keras.datasets import cifar10
from keras.datasets import mnist

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras import backend as K

import numpy as np
import resnet

def generateBigDB(objectiveNumberOfSamples):
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print('x_train Before enhancement ' + str(len(x_train)))
    print('y_train Before enhancement ' + str(len(y_train)))

    # problem was here- change of labels from 3 to .0 .0 .1 .0 .0 .0 .0 .0 .0 .0
    # y_train = np_utils.to_categorical(y_train, num_classes)
    # y_test = np_utils.to_categorical(y_test, num_classes)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(x_train)

    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=10000):
        if len(x_train) >= objectiveNumberOfSamples:
            break
        x_train = np.append(x_train,x_batch,0)
        y_train = np.append(y_train,y_batch,0)
        print(len(x_train))
        print(len(y_train))

    print('x_train After enhancement ' + str(len(x_train)))
    print('y_train After enhancement ' + str(len(y_train)))
    y_train.astype(int)
    return (x_train, y_train), (x_test, y_test)

def saveData(model, X_train, y_train, X_test, y_test, nb_classes):
    for layer_index in range(1, len(model.layers), 1):
        if layer_index % 10 == 0:
            print('Layer index is ' + str(layer_index))
        shouldIContinue = True
        if 'conv2d' in model.layers[layer_index].name and layer_index>30:
           shouldIContinue = False
        if 'add' in model.layers[layer_index].name:
           shouldIContinue = False
        if 'flatten' in model.layers[layer_index].name:
           shouldIContinue = False
        if 'dense' in model.layers[layer_index].name:
            shouldIContinue = False
        if shouldIContinue:
            continue
        if layer_index % 20 == 0:
            print("think about it")
        directory = 'C:\cifar//residual//layer_output//layer_' + str(layer_index) + '_' + model.layers[layer_index].name
        if os.path.exists(directory):
            print('Folder already exists')
        else :
            os.mkdir(directory)

        get_k_layer_output = K.function([model.layers[0].input],
                                        [model.layers[layer_index].output])
        batchSize = 2000
        with open(directory + '//trainingData.txt', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            for startingImageIndex in range(int(len(X_train) / batchSize)):
                sample = X_train[startingImageIndex*batchSize:startingImageIndex*batchSize + batchSize]
                layer_output = get_k_layer_output([sample])[0]
                for imageIndex in range(len(sample)):
                    output = layer_output[imageIndex].flatten()
                    index = np.argmax(output)
                    row = [0] * len(output)
                    row[index] = 1
                    spamwriter.writerow(output)
        with open(directory + '//trainingLabel.txt', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            for label in y_train:
                row = [0] * nb_classes
                row[label[0]] = 1
                spamwriter.writerow(row)
        with open(directory + '//testingData.txt', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            for startingImageIndex in range(int(len(X_test) / batchSize)):
                sample = X_test[startingImageIndex:startingImageIndex + batchSize]
                layer_output = get_k_layer_output([sample])[0]
                for imageIndex in range(len(sample)):
                    output = layer_output[imageIndex].flatten()
                    index = np.argmax(output)
                    row = [0] * len(output)
                    row[index] = 1
                    spamwriter.writerow(output)
        with open(directory + '//testingLabel.txt', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            for label in y_test:
                row = [0] * nb_classes
                row[label[0]] = 1
                spamwriter.writerow(row)



def main():

    print(K.tensorflow_backend._get_available_gpus())

    saveOutput = False
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('resnet18_cifar10.csv')

    batch_size = 32
    nb_classes = 10
    nb_epoch = 100

    # input image dimensions
    img_rows, img_cols = 32, 32
    # The CIFAR10 images are RGB.
    img_channels = 3

    objectiveNumberOfSamples = 100000
    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = generateBigDB(objectiveNumberOfSamples)
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()

    fracOfSamples = 1.0
    fracOfFeatures = 1.0

    X_train = X_train[:int(len(X_train)*fracOfSamples),:,:,:]
    y_train = y_train[:int(len(y_train)*fracOfSamples)]
    X_test = X_test[:int(len(X_test)*fracOfSamples),:,:,:]
    y_test = y_test[:int(len(y_test)*fracOfSamples)]

    if saveOutput:
        #directory = 'C:\ProgrammingOriented//projects//safe//'
        directory = 'C:\cifar//'


        suf1 = 'residual'
        suf2 = 'residual//layer_output'
        suf3 = 'residual//layer_output//layer_0'

        if os.path.exists(directory + suf1):
            print('Folder already exists')
        else :
            os.mkdir(directory + suf1)

        if os.path.exists(directory + suf2):
            print('Folder already exists')
        else :
            os.mkdir(directory + suf2)

        if os.path.exists(directory + suf3):
            print('Folder already exists')
        else:
            os.mkdir(directory + suf3)

        print('Layer index is 0')

        with open('C:\cifar//residual//layer_output//layer_' + str(0) + '//trainingData.txt', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            count = 0
            for label in X_train:
                count += 1
                print(str(count) + ' out of ' + str(len(X_train)))
                output = np.array(label).flatten()
                output = output[:int(fracOfFeatures*len(output))]
                output -= output.min()
                if output.max() != 0:
                    output = output / (output.max())
                spamwriter.writerow(output)
        with open('C:\cifar//residual//layer_output//layer_' + str(0) + '//trainingLabel.txt', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            for label in y_train:
                row = [0] * nb_classes
                row[label[0]] = 1
                spamwriter.writerow(row)
        with open('C:\cifar//residual//layer_output//layer_' + str(0) + '//testingData.txt', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            for label in X_test:
                output = np.array(label).flatten()
                output = output[:int(fracOfFeatures*len(output))]
                output -= output.min()
                if output.max != 0.0:
                    output = output / (output.max())
                spamwriter.writerow(output)
        with open('C:\cifar//residual//layer_output//layer_' + str(0) + '//testingLabel.txt', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            for label in y_test:
                row = [0] * nb_classes
                row[label[0]] = 1
                spamwriter.writerow(row)
    # Convert class vectors to binary class matrices.

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.


    model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)




    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=False,
              callbacks=[lr_reducer, early_stopper, csv_logger])
    '''
    fitSomeMore = input("fit some more?")
    while bool(fitSomeMore):
        nb_epoch1 = int(input("how many more epochs?"))
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch1,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper, csv_logger])
        fitSomeMore = input("fit some more?")

    SaveResults = input("Continue to Save results?")
    if bool(SaveResults):
        saveData(model, X_train, y_train, X_test, y_test, nb_classes)
    '''
    saveData(model, X_train, y_train, X_test, y_test, nb_classes)

main()