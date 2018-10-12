import dataset
from classifier import Classifier

print("Loading dataset...")

trainSet, validationSet, testSet = dataset.loadDataset()

print("Dataset loaded!")

# class1 = Classifier(0.05, 0, 100, 50, 0.95, shuffling = True)
# class1.grid_search(trainSet["data"], trainSet["one_hot"], validationSet, [0.05, 0.075, 0.09, 0.01, 0.025], [0.0001], [0.95, 0.9, 0.85])


#class1 = Classifier(0.05, 0, 100, 500, 0.95, shuffling = True, SVM = False)
class1 = Classifier(0.01, 0, 100, 50, 0.95, shuffling = True, SVM = False)
class1.fit(trainSet["data"], trainSet["one_hot"], validationSet = validationSet)

print("Final accuracy:")
print(" " + str(class1.computeAccuracy(testSet["data"], testSet["labels"])))

# class2 = Classifier(0.01, 0, 100, 50, 0.95, shuffling = True, SVM = True)
# class2.fit(trainSet["data"], trainSet["one_hot"], validationSet = validationSet)
#
# print("Final accuracy:")
# print(" " + str(class2.computeAccuracy(testSet["data"], testSet["labels"])))
