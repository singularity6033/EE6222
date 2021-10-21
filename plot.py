import pickle

import numpy as np
from matplotlib import pyplot as plt

dataset_names = ["abalone", "bank", "balloons", "car", "credit",
                 "ecoli", "flags", "glass", "letter", "iris"]
ACC = pickle.loads(open('results.pickle', "rb").read())
train_acc = pickle.loads(open('train_acc.pickle', "rb").read())
test_acc = pickle.loads(open('test_acc.pickle', "rb").read())
ACC = np.mean(ACC, axis=-1)

plt.figure(figsize=(9, 9))
plt.style.use("ggplot")
plt.plot(dataset_names, ACC[:, 0], marker='o')
plt.plot(dataset_names, ACC[:, 1], marker='x')
plt.grid()
plt.title("Effect of direct links from the input layer to the output layer")
plt.ylabel("Mean Accuracy after 4-fold cross-validation")
plt.legend(['with link', 'without link'], loc=1)
plt.xticks(rotation=20)
plt.savefig('output/1.png')
plt.close()

plt.figure(figsize=(9, 9))
plt.style.use("ggplot")
plt.plot(dataset_names, ACC[:, 2], marker='.')
plt.plot(dataset_names, ACC[:, 3])
plt.grid()
plt.title("Performance comparisons 2 activation functions (sigmoid vs hardlim)")
plt.ylabel("Mean Accuracy after 4-fold cross-validation")
plt.legend(['sigmoid', 'hardlim'], loc=1)
plt.xticks(rotation=20)
plt.savefig('output/2.png')
plt.close()

plt.figure(figsize=(9, 9))
plt.style.use("ggplot")
plt.plot(dataset_names, ACC[:, 4], marker='v')
plt.plot(dataset_names, ACC[:, 5], marker='*')
plt.grid()
plt.title("Performance of Moore-Penrose pseudoinverse and ridge regression")
plt.ylabel("Mean Accuracy after 4-fold cross-validation")
plt.legend(['Moore-Penrose pseudoinverse', 'ridge regression'], loc=1)
plt.xticks(rotation=20)
plt.savefig('output/3.png')
plt.close()
