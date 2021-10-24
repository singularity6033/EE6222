import pickle

import numpy as np
from matplotlib import pyplot as plt

dataset_names = ["abalone", "car", "ringnorm", "ecoli", "miniboone", "semeion", "iris", "magic", "ozone", "zoo"]

ACC = pickle.loads(open('activation_acc.pickle', "rb").read())
train_acc = pickle.loads(open('activation_train_acc.pickle', "rb").read())
test_acc = pickle.loads(open('activation_test_acc.pickle', "rb").read())

train_acc_mean = np.mean(train_acc, axis=-1)
test_acc_mean = np.mean(test_acc, axis=-1)
ACC_mean = np.mean(ACC, axis=-1)
ACC_var = np.std(ACC, axis=-1)

x = np.arange(len(dataset_names))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots(2, 1, figsize=(13, 25))
rects1 = ax[0].bar(x - 3*width, train_acc_mean[:, 0], width, label='relu')
rects2 = ax[0].bar(x - 2*width, train_acc_mean[:, 1], width, label='sigmoid')
rects3 = ax[0].bar(x - width, train_acc_mean[:, 2], width, label='radbas')
rects4 = ax[0].bar(x, train_acc_mean[:, 3], width, label='sine')
rects5 = ax[0].bar(x + width, train_acc_mean[:, 4], width, label='hardlim')
rects6 = ax[0].bar(x + 2*width, train_acc_mean[:, 5], width, label='tribas')
ax[0].set_ylabel('mean accuracy', fontsize=20)
ax[0].set_ylim(0.5, 1)
ax[0].set_title('comparisons among different choices of activation functions in train set', fontsize=20)
ax[0].set_xticks(x)
ax[0].set_xticklabels(dataset_names, rotation=20, fontsize=20)
ax[0].legend(fontsize=20)
ax[0].tick_params(labelsize=20)
rects11 = ax[1].bar(x - 3*width, test_acc_mean[:, 0], width, label='relu')
rects22 = ax[1].bar(x - 2*width, test_acc_mean[:, 1], width, label='sigmoid')
rects33 = ax[1].bar(x - 1*width, test_acc_mean[:, 2], width, label='radbas')
rects44 = ax[1].bar(x, test_acc_mean[:, 3], width, label='sine')
rects55 = ax[1].bar(x + 1*width, test_acc_mean[:, 4], width, label='hardlim')
rects66 = ax[1].bar(x + 2*width, test_acc_mean[:, 5], width, label='tribas')
ax[1].set_ylabel('mean accuracy', fontsize=20)
ax[1].set_title('comparisons among different choices of activation functions in test set', fontsize=20)
ax[1].set_xticks(x)
ax[1].set_ylim(0.5, 1)
ax[1].set_xticklabels(dataset_names, fontsize=20, rotation=20)
ax[1].legend(fontsize=20)
ax[1].tick_params(labelsize=20)
fig.tight_layout()
# plt.show()

plt.savefig('output/3.png')
plt.close()
