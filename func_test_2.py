import os
import numpy as np
import random
import pickle
from function_new import RVFL_train_val
import h5py
from option import option as op

dataset_names = ["abalone", "car", "ringnorm", "ecoli", "miniboone", "semeion", "iris", "magic", "ozone", "zoo"]
num_dataset = len(dataset_names)

# # Look at the documentation of RVFL_train_val function file
option1 = [op()] * num_dataset
option2 = [op()] * num_dataset
S = np.linspace(-5, 5, 21)
epochs = len(range(-5, 15)) * len(range(3, 204, 20))
train_acc = np.zeros((10, 2, epochs))
test_acc = np.zeros((10, 2, epochs))
ACC_CV = np.zeros([num_dataset, 2, 4])

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
    MAX_acc = np.zeros([2, 1])
    Best_N = np.zeros([2, 1]).astype(np.int32)
    Best_C = np.zeros([2, 1])
    Best_S = np.zeros([2, 1])
    count = 0
    for N in range(3, 204, 20):
        for C in range(-5, 15):
            Scale = 1

            # using Moore-Penrose pseudoinverse
            option1[i].mode = 2
            option1[i].N = N
            option1[i].Scale = Scale
            option1[i].C = 2 ** C
            option1[i].Scalemode = 3
            option1[i].bias = 1
            option1[i].link = 1

            # using ridge regression (or regularized least square solutions)
            option2[i].mode = 1
            option2[i].N = N
            option2[i].Scale = Scale
            option2[i].C = 2 ** C
            option2[i].Scalemode = 3
            option2[i].bias = 1
            option2[i].link = 1

            train_accuracy1, test_accuracy1 = RVFL_train_val(trainX, trainY, testX, testY, option1[i])
            train_accuracy2, test_accuracy2 = RVFL_train_val(trainX, trainY, testX, testY, option2[i])

            train_acc[i, 0, count] = train_accuracy1
            train_acc[i, 1, count] = train_accuracy2
            test_acc[i, 0, count] = train_accuracy1
            test_acc[i, 1, count] = train_accuracy2

            count += 1
            # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
            if test_accuracy1 > MAX_acc[0]:
                MAX_acc[0] = test_accuracy1
                Best_N[0] = N
                Best_C[0] = C

            if test_accuracy2 > MAX_acc[1]:
                MAX_acc[1] = test_accuracy2
                Best_N[1] = N
                Best_C[1] = C

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

        option1[i].mode = 2
        option1[i].N = Best_N[0, 0]
        option1[i].Scale = 1
        option1[i].Scalemode = 3
        option1[i].C = 2 ** C
        option1[i].bias = 1
        option1[i].link = 1

        option1[i].mode = 1
        option2[i].N = Best_N[1, 0]
        option2[i].Scale = 1
        option2[i].C = 2 ** C
        option2[i].Scalemode = 3
        option2[i].bias = 1
        option2[i].link = 1

        train_accuracy1, ACC_CV[i, 0, j] = RVFL_train_val(trainX, trainY, testX, testY, option1[i])
        train_accuracy2, ACC_CV[i, 1, j] = RVFL_train_val(trainX, trainY, testX, testY, option2[i])

print("[INFO] saving ACC...")
f = open('func_acc.pickle', "wb")
f.write(pickle.dumps(ACC_CV))
f.close()

print("[INFO] saving train_acc...")
f = open('func_train_acc.pickle', "wb")
f.write(pickle.dumps(train_acc))
f.close()

print("[INFO] saving test_acc...")
f = open('func_test_acc.pickle', "wb")
f.write(pickle.dumps(test_acc))
f.close()