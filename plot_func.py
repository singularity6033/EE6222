import pickle

import numpy as np
from matplotlib import pyplot as plt

dataset_names = ["abalone", "car", "ringnorm", "ecoli", "miniboone", "semeion", "iris", "magic", "ozone", "zoo"]

ACC = pickle.loads(open('func_acc.pickle', "rb").read())
train_acc = pickle.loads(open('func_train_acc.pickle', "rb").read())
test_acc = pickle.loads(open('func_test_acc.pickle', "rb").read())

train_acc_mean = np.mean(train_acc, axis=-1)
test_acc_mean = np.mean(test_acc, axis=-1)
ACC_mean = np.mean(ACC, axis=-1)
ACC_var = np.std(ACC, axis=-1)

x = np.arange(len(dataset_names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(2, 1, figsize=(13, 25))
rects1 = ax[0].bar(x - width/2, train_acc_mean[:, 0], width, label='Moore-Penrose pseudoinverse')
rects2 = ax[0].bar(x + width/2, train_acc_mean[:, 1], width, label='regularized least square')
ax[0].set_ylabel('mean accuracy', fontsize=20)
ax[0].set_title('comparisons between two close form solutions in train set', fontsize=20)
ax[0].set_xticks(x)
ax[0].set_xticklabels(dataset_names, rotation=20, fontsize=20)
ax[0].legend(fontsize=20)
ax[0].bar_label(rects1, padding=3, fontsize=10)
ax[0].bar_label(rects2, padding=3, fontsize=10)
ax[0].tick_params(labelsize=20)
rects11 = ax[1].bar(x - width/2, test_acc_mean[:, 0], width, label='Moore-Penrose pseudoinverse')
rects22 = ax[1].bar(x + width/2, test_acc_mean[:, 1], width, label='regularized least square')
ax[1].set_ylabel('mean accuracy', fontsize=20)
ax[1].set_title('comparisons between two close form solutions in test set', fontsize=20)
ax[1].set_xticks(x)
ax[1].set_xticklabels(dataset_names, fontsize=20, rotation=20)
ax[1].legend(fontsize=20)
ax[1].bar_label(rects11, padding=3, fontsize=10)
ax[1].bar_label(rects22, padding=3, fontsize=10)
ax[1].tick_params(labelsize=20)
fig.tight_layout()
# plt.show()

plt.savefig('output/2.png')
plt.close()
