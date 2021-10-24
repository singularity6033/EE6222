import os

import numpy as np
import pickle
from function_new import RVFL_train_val
import h5py
from option import option as op

dataset_names = ["abalone", "car", "ringnorm", "ecoli", "miniboone", "semeion", "iris", "magic", "ozone", "zoo"]
num_dataset = len(dataset_names)

# # Look at the documentation of RVFL_train_val function file
option1 = [op()] * num_dataset
option2 = [op()] * num_dataset
option3 = [op()] * num_dataset
option4 = [op()] * num_dataset
option5 = [op()] * num_dataset
option6 = [op()] * num_dataset

epochs = len(range(0, np.linspace(-5, 5, 21).size)) * len(range(3, 204, 20))
train_acc = np.zeros((10, 6, epochs))
test_acc = np.zeros((10, 6, epochs))
ACC_CV = np.zeros([num_dataset, 6, 4])

for i, dataset_name in enumerate(dataset_names):
    print("[INFO] processing dataset {}...".format(dataset_name))
    temp = h5py.File(os.path.sep.join(["UCI data python", dataset_name + "_R.mat"]))
    data = np.array(temp['data']).T

    data = data[:, 1:]
    dataX = data[:, 0:-1]
    # do normalization for each feature

    dataX_mean = np.mean(dataX, axis=0)
    dataX_std = np.std(dataX, axis=0)
    dataX = (dataX - dataX_mean) / dataX_std
    dataY = data[:, -1]
    dataY = np.expand_dims(dataY, 1)

    temp = h5py.File(os.path.sep.join(["UCI data python", dataset_name + "_conxuntos.mat"]))
    index1 = np.array(temp['index1']).astype(np.int32) - 1
    index2 = np.array(temp['index2']).astype(np.int32) - 1
    index1 = np.squeeze(index1, axis=1)
    index2 = np.squeeze(index2, axis=1)

    trainX = dataX[index1, :]
    trainY = dataY[index1, :]
    testX = dataX[index2, :]
    testY = dataY[index2, :]
    MAX_acc = np.zeros([6, 1])
    Best_N = np.zeros([6, 1]).astype(np.int32)
    Best_C = np.zeros([6, 1])
    Best_S = np.zeros([6, 1])
    count = 0
    for N in range(3, 204, 20):
        for C in range(-5, 15):
            Scale = 1

            # using activation function relu
            option1[i].ActivationFunction = 'relu'
            option1[i].N = N
            option1[i].C = 2 ** C
            option1[i].Scale = Scale
            option1[i].Scalemode = 3
            option1[i].bias = 1
            option1[i].link = 1

            # using activation function sigmoid
            option2[i].ActivationFunction = 'sigmoid'
            option2[i].N = N
            option2[i].C = 2 ** C
            option2[i].Scale = Scale
            option2[i].Scalemode = 3
            option2[i].bias = 1
            option2[i].link = 1

            # using activation function radbas
            option3[i].ActivationFunction = 'radbas'
            option3[i].N = N
            option3[i].C = 2 ** C
            option3[i].Scale = Scale
            option3[i].Scalemode = 3
            option3[i].bias = 1
            option3[i].link = 1

            # using activation function sine
            option4[i].ActivationFunction = 'sine'
            option4[i].N = N
            option4[i].C = 2 ** C
            option4[i].Scale = Scale
            option4[i].Scalemode = 3
            option4[i].bias = 1
            option4[i].link = 1

            # using activation function hardlim
            option5[i].ActivationFunction = 'hardlim'
            option5[i].N = N
            option5[i].C = 2 ** C
            option5[i].Scale = Scale
            option5[i].Scalemode = 3
            option5[i].bias = 1
            option5[i].link = 1

            # using activation function tribas
            option6[i].ActivationFunction = 'tribas'
            option6[i].N = N
            option6[i].C = 2 ** C
            option6[i].Scale = Scale
            option6[i].Scalemode = 3
            option6[i].bias = 1
            option6[i].link = 1

            train_accuracy1, test_accuracy1 = RVFL_train_val(trainX, trainY, testX, testY, option1[i])
            train_accuracy2, test_accuracy2 = RVFL_train_val(trainX, trainY, testX, testY, option2[i])
            train_accuracy3, test_accuracy3 = RVFL_train_val(trainX, trainY, testX, testY, option3[i])
            train_accuracy4, test_accuracy4 = RVFL_train_val(trainX, trainY, testX, testY, option4[i])
            train_accuracy5, test_accuracy5 = RVFL_train_val(trainX, trainY, testX, testY, option5[i])
            train_accuracy6, test_accuracy6 = RVFL_train_val(trainX, trainY, testX, testY, option6[i])

            train_acc[i, 0, count] = train_accuracy1
            train_acc[i, 1, count] = train_accuracy2
            train_acc[i, 2, count] = train_accuracy3
            train_acc[i, 3, count] = train_accuracy4
            train_acc[i, 4, count] = train_accuracy5
            train_acc[i, 5, count] = train_accuracy6
            test_acc[i, 0, count] = train_accuracy1
            test_acc[i, 1, count] = train_accuracy2
            test_acc[i, 2, count] = train_accuracy3
            test_acc[i, 3, count] = train_accuracy4
            test_acc[i, 4, count] = train_accuracy5
            test_acc[i, 5, count] = train_accuracy6
            count += 1
            # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
            if test_accuracy1 > MAX_acc[0]:
                MAX_acc[0] = test_accuracy1
                Best_N[0] = N
                Best_C[0] = C
                Best_S[0] = Scale

            if test_accuracy2 > MAX_acc[1]:
                MAX_acc[1] = test_accuracy2
                Best_N[1] = N
                Best_C[1] = C
                Best_S[1] = Scale

            if test_accuracy3 > MAX_acc[2]:
                MAX_acc[2] = test_accuracy3
                Best_N[2] = N
                Best_C[2] = C
                Best_S[2] = Scale

            if test_accuracy4 > MAX_acc[3]:
                MAX_acc[3] = test_accuracy4
                Best_N[3] = N
                Best_C[3] = C
                Best_S[3] = Scale

            if test_accuracy5 > MAX_acc[4]:
                MAX_acc[4] = test_accuracy5
                Best_N[4] = N
                Best_C[4] = C
                Best_S[4] = Scale

            if test_accuracy6 > MAX_acc[5]:
                MAX_acc[5] = test_accuracy6
                Best_N[5] = N
                Best_C[5] = C
                Best_S[5] = Scale

    temp = h5py.File(os.path.sep.join(["UCI data python", dataset_name + "_conxuntos_kfold.mat"]))
    index = []

    for j in range(8):
        index_temp = np.array([temp[element[j]][:] for element in temp['index']]).astype(np.int32) - 1
        index_temp = np.squeeze(index_temp, axis=0)
        index_temp = np.squeeze(index_temp, axis=1)
        index.append(index_temp)

    for j in range(4):
        trainX = dataX[index[2 * j], :]
        trainY = dataY[index[2 * j], :]
        testX = dataX[index[2 * j + 1], :]
        testY = dataY[index[2 * j + 1], :]

        option1[i].N = Best_N[0, 0]
        option1[i].C = 2 ** Best_C[0, 0]
        option1[i].Scale = Best_S[0, 0]
        option1[i].Scalemode = 3
        option1[i].bias = 1
        option1[i].link = 1
        option1[i].ActivationFunction = 'relu'

        option2[i].N = Best_N[1, 0]
        option2[i].C = 2 ** Best_C[1, 0]
        option2[i].Scale = Best_S[1, 0]
        option2[i].Scalemode = 3
        option2[i].bias = 1
        option2[i].link = 1
        option2[i].ActivationFunction = 'sigmoid'

        option3[i].N = Best_N[2, 0]
        option3[i].C = 2 ** Best_C[2, 0]
        option3[i].Scale = Best_S[2, 0]
        option3[i].Scalemode = 3
        option3[i].bias = 1
        option3[i].link = 1
        option3[i].ActivationFunction = 'radbas'

        option4[i].N = Best_N[3, 0]
        option4[i].C = 2 ** Best_C[3, 0]
        option4[i].Scale = Best_S[3, 0]
        option4[i].Scalemode = 3
        option4[i].bias = 1
        option4[i].link = 1
        option4[i].ActivationFunction = 'sine'

        option5[i].N = Best_N[4, 0]
        option5[i].C = 2 ** Best_C[4, 0]
        option5[i].Scale = Best_S[4, 0]
        option5[i].Scalemode = 3
        option5[i].bias = 1
        option5[i].link = 1
        option5[i].ActivationFunction = 'hardlim'

        option6[i].N = Best_N[5, 0]
        option6[i].C = 2 ** Best_C[5, 0]
        option6[i].Scale = Best_S[5, 0]
        option6[i].Scalemode = 3
        option6[i].bias = 1
        option6[i].link = 1
        option6[i].ActivationFunction = 'tribas'

        train_accuracy1, ACC_CV[i, 0, j] = RVFL_train_val(trainX, trainY, testX, testY, option1[i])
        train_accuracy2, ACC_CV[i, 1, j] = RVFL_train_val(trainX, trainY, testX, testY, option2[i])
        train_accuracy3, ACC_CV[i, 2, j] = RVFL_train_val(trainX, trainY, testX, testY, option3[i])
        train_accuracy4, ACC_CV[i, 3, j] = RVFL_train_val(trainX, trainY, testX, testY, option4[i])
        train_accuracy5, ACC_CV[i, 4, j] = RVFL_train_val(trainX, trainY, testX, testY, option5[i])
        train_accuracy6, ACC_CV[i, 5, j] = RVFL_train_val(trainX, trainY, testX, testY, option6[i])

print("[INFO] saving ACC...")
f = open('activation_acc.pickle', "wb")
f.write(pickle.dumps(ACC_CV))
f.close()

print("[INFO] saving train_acc...")
f = open('activation_train_acc.pickle', "wb")
f.write(pickle.dumps(train_acc))
f.close()

print("[INFO] saving test_acc...")
f = open('activation_test_acc.pickle', "wb")
f.write(pickle.dumps(test_acc))
f.close()
