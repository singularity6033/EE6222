import pickle

import numpy as np
from matplotlib import pyplot as plt

dataset_names = ["abalone", "car", "ringnorm", "ecoli", "miniboone", "semeion", "iris", "magic", "ozone", "zoo"]

ACC = pickle.loads(open('link_acc.pickle', "rb").read())
train_acc = pickle.loads(open('link_train_acc.pickle', "rb").read())
test_acc = pickle.loads(open('link_test_acc.pickle', "rb").read())

train_acc_mean = np.mean(train_acc, axis=-1)
test_acc_mean = np.mean(test_acc, axis=-1)
ACC_mean = np.mean(ACC, axis=-1)
ACC_var = np.std(ACC, axis=-1)

x = np.arange(len(dataset_names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(1, 2, figsize=(25, 10))
rects1 = ax[0].bar(x - width/2, train_acc_mean[:, 0], width, label='Without Link')
rects2 = ax[0].bar(x + width/2, train_acc_mean[:, 1], width, label='With Link')
ax[0].set_ylabel('mean accuracy', fontsize=10)
ax[0].set_title('performance comparisons 2 activation functions (sigmoid vs hardlim) in train set', fontsize=20)
ax[0].set_xticks(x)
ax[0].set_xticklabels(dataset_names, rotation=20, fontsize=20)
ax[0].legend(fontsize=20)
ax[0].bar_label(rects1, padding=3, fontsize=20)
ax[0].bar_label(rects2, padding=3, fontsize=20)
ax[0].tick_params(labelsize=20)
rects11 = ax[1].bar(x - width/2, test_acc_mean[:, 0], width, label='Without Link')
rects22 = ax[1].bar(x + width/2, test_acc_mean[:, 1], width, label='With Link')
ax[1].set_ylabel('mean accuracy', fontsize=10)
ax[1].set_title('performance comparisons 2 activation functions (sigmoid vs hardlim) in test set', fontsize=20)
ax[1].set_xticks(x)
ax[1].set_xticklabels(dataset_names, fontsize=20, rotation=20)
ax[1].legend(fontsize=20)
ax[1].bar_label(rects11, padding=3, fontsize=20)
ax[1].bar_label(rects22, padding=3, fontsize=20)
ax[1].tick_params(labelsize=20)
fig.tight_layout()
plt.show()


#
# plt.figure(figsize=(9, 9))
# plt.style.use("ggplot")
# plt.plot(dataset_names, ACC[:, 4], marker='v')
# plt.plot(dataset_names, ACC[:, 5], marker='*')
# plt.grid()
# plt.title("Performance of Moore-Penrose pseudoinverse and ridge regression")
# plt.ylabel("Mean Accuracy after 4-fold cross-validation")
# plt.legend(['Moore-Penrose pseudoinverse', 'ridge regression'], loc=1)
# plt.xticks(rotation=20)
# plt.savefig('output/3.png')
# plt.close()
