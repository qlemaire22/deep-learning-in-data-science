import pickle
import numpy as np
import constants
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def createOneHotLabels(labels):
    one_hot_labels = np.zeros((constants.N, constants.K))
    for i in range(len(labels)):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels

def loadBatch(filename):
    data = np.zeros((constants.N, constants.d))
    labels = np.zeros((constants.N, 1))
    one_hot_labels = np.zeros((constants.N, constants.K))

    dict = unpickle(filename)
    data = dict[bytes("data", 'utf-8')] / 255.0
    labels = np.array(dict[bytes("labels", 'utf-8')])
    one_hot_labels = createOneHotLabels(labels)

    return data.T, one_hot_labels.T, labels

def loadDataset():
    trainSet = {}
    testSet = {}
    validationSet = {}

    for i in [1, 3, 4, 5]:
        t1, t2, t3 = loadBatch("../dataset/data_batch_" + str(i))
        if i == 1:
            trainSet["data"] = t1
            trainSet["one_hot"] = t2
            trainSet["labels"] = t3
        else:
            trainSet["data"] = np.column_stack((trainSet["data"], t1))
            trainSet["one_hot"] = np.column_stack((trainSet["one_hot"], t2))
            trainSet["labels"] = np.append(trainSet["labels"], t3)

    a, b, c = loadBatch("../dataset/data_batch_2")

    validationSet["data"], validationSet["one_hot"], validationSet["labels"] = a[:, :1000], b[:, :1000], c[:1000]

    trainSet["data"] = np.column_stack((trainSet["data"], a[:, 1000:]))
    trainSet["one_hot"] = np.column_stack((trainSet["one_hot"], b[:, 1000:]))
    trainSet["labels"] = np.append(trainSet["labels"], c[1000:])
    testSet["data"], testSet["one_hot"], testSet["labels"] = loadBatch("../dataset/test_batch")

    temp = np.copy(trainSet["data"]).reshape((32, 32, 3, 49000), order='F')
    temp = np.flip(temp, 0)
    temp = temp.reshape((3072, 49000), order='F')

    trainSet["data"] = np.column_stack((trainSet["data"], temp))
    trainSet["one_hot"] = np.column_stack((trainSet["one_hot"], trainSet["one_hot"]))
    trainSet["labels"] = np.append(trainSet["labels"], trainSet["labels"])

    return normalization(trainSet, validationSet, testSet)

def plotImage(image):
    image = np.rot90(np.reshape(image, (32, 32, 3), order='F'), k=3)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def normalization(trainSet, validationSet, testSet):
    mean = np.mean(trainSet["data"], axis=1)
    mean = mean[:, np.newaxis]
    trainSet["data"] = trainSet["data"] - mean
    validationSet["data"] = validationSet["data"] - mean
    testSet["data"] = testSet["data"] - mean
    return trainSet, validationSet, testSet
