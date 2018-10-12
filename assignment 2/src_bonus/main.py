import dataset
from classifier import Classifier

print("Loading dataset...")
trainSet, validationSet, testSet = dataset.loadDataset()
print("Dataset loaded!")


class1 = Classifier(leaky_RELU=True)

class1.fit(trainSet["data"], trainSet["one_hot"], validationSet = validationSet, graphics=False)
#class1.grid_search(trainSet["data"], trainSet["one_hot"], validationSet, [0.1, 0.01, 0.001, 0.0001, 0.9, 0.00001, 0.5], [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], [0.95])
#class1.random_search(trainSet["data"][:, :5000], trainSet["one_hot"][:, :5000], validationSet, 100, [0.001, 0.02], [5.0e-05, .001])

print("Final accuracy:")
print(" " + str(class1.computeAccuracy(testSet["data"], testSet["labels"])))
