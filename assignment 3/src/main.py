from network import ConvNet
import numpy as np
from dataset import getDataset

dataset = getDataset()

print(dataset["balance_train"])

conv = ConvNet(dataset)
conv.fit(dataset["one_hot_train"], dataset["one_hot_label_train"], dataset["one_hot_validation"], dataset["one_hot_label_validation"], graphics=True)

print("Final accuracy:")
print(" " + str(conv.computeAccuracy(dataset["one_hot_validation"], dataset["labels_validation"])))
print("Friends:")
conv.classify(dataset["one_hot_friends"])
#
# F = np.zeros((4, 2, 2))
# # print(F[:, :, 0])
# # print(F[:, :, 0].shape)
# F[:, :, 0] = [[1, 2], [3, 4], [5, 6], [7, 8]]
# F[:, :, 0] = [[1, 2], [3, 4], [5, 6], [7, 8]]
# # print(F[:, :, 0])
# # print(F[:, :, 1])
# # print(F)
#
# MF = conv.makeMFMatrix(F, 4)
#
#
# X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
#
# # print(X)
# MX = conv.makeMXMatrix(X, 4, 2, 2)
#
# VF = np.zeros((16, 1))
# for i in range(2):
#     VF[i*8:i*8+8, :] = (F[:, :, i].reshape((8, 1), order='F'))
#
# print(np.dot(MX, VF))
#
# VX = np.zeros((16, 1))
# for i in range(4):
#     VX[i*4:i*4+4, :] = (X[:, i].reshape((4, 1), order='F'))
#
#
# print(np.dot(MF, VX))
#
# print(np.dot(MX, conv.vectF(F)))
# print(np.dot(MF, conv.vectX(X)))
