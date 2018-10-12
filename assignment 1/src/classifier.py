import numpy as np
import constants
from math import floor, sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt


class Classifier():
    def __init__(self, learning_rate, regularization_term, batch_size, n_epochs, weight_decay, shuffling=False, SVM=False):
        self.W = np.zeros((constants.K, constants.d))
        self.b = np.zeros((constants.K, 1))

        self.eta = learning_rate
        self.lambda_reg = regularization_term
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.shuffling = shuffling
        self.SVM = SVM

        np.random.seed(1)

        self.initialization()

    def initialization(self):
        mu = 0
        sigma = sqrt(2) / sqrt(constants.d)

        self.W = np.random.normal(mu, sigma, (constants.K, constants.d))
        self.b = np.random.normal(mu, sigma, (constants.K, 1))

    def evaluateClassifier(self, X, W, b):
        s = np.dot(W, X) + b
        P = self.softmax(s)
        assert(P.shape == (constants.K, X.shape[1]))
        return P

    def softmax(self, x):
        r = np.exp(x) / sum(np.exp(x))
        return r

    def computeCost(self, X, Y, W, b):
        regularization = self.lambda_reg * np.sum(np.square(W))
        loss_sum = 0
        for i in range(X.shape[1]):
            x = np.zeros((constants.d, 1))
            y = np.zeros((constants.K, 1))
            x = X[:, [i]]
            y = Y[:, [i]]
            if (self.SVM):
                loss_sum += self.svm_loss(x, y, W=W, b=b)
            else:
                loss_sum += self.cross_entropy(x, y, W=W, b=b)


        loss_sum /= X.shape[1]
        final = loss_sum + regularization
        assert(len(final) == 1)
        return final

    def cross_entropy(self, x, y, W, b):
        l = - np.log(np.dot(y.T, self.evaluateClassifier(x, W=W, b=b)))[0]
        return l

    def svm_loss(self, x, y, W, b):
        s = np.dot(W, x) + b
        l = 0
        y_int = np.where(y.T[0] == 1)[0][0]
        for j in range(constants.K):
            if j != y_int:
                l += max(0, s[j] - s[y_int] + 1)
        return l

    def computeAccuracy(self, X, Y):
        acc = 0
        for i in range(X.shape[1]):
            P = self.evaluateClassifier(X[:, [i]], self.W, self.b)
            label = np.argmax(P)
            if label == Y[i]:
                acc += 1
        acc /= X.shape[1]
        return acc

    def compute_gradients(self, X, Y, P, W):
        G = -(Y - P.T).T
        return (G@X) / X.shape[0] + 2 * self.lambda_reg * W, np.mean(G, axis=-1, keepdims=True)

    def compute_gradients_SVM(self, X, Y, W, b):
        n = X.shape[1]
        gradW = np.zeros((constants.K, constants.d))
        gradb = np.zeros((constants.K, 1))

        for i in range(n):
            x = X[:, i]
            y = Y[:, [i]]
            y_int = np.where(Y[:, [i]].T[0] == 1)[0][0]
            s = np.dot(W, X[:, [i]]) + b
            for j in range(constants.K):
                if j != y_int:
                    if max(0, s[j] - s[y_int] + 1) != 0:
                        gradW[j] += x
                        gradW[y_int] += -x
                        gradb[j, 0] += 1
                        gradb[y_int, 0] += -1

        gradW /= n
        gradW += self.lambda_reg * W
        gradb /= n
        return gradW, gradb

    def unison_shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def fit(self, X, Y, validationSet=[], graphics=False):
        n = X.shape[1]
        costsTraining = []
        costsValidation = []
        bestW = np.copy(self.W)
        bestb = np.copy(self.b)
        bestVal = self.computeCost(
            validationSet["data"], validationSet["one_hot"], self.W, self.b)[0]
        bestEpoch = 0

        for i in tqdm(range(self.n_epochs)):
            n_batch = floor(n / self.batch_size)

            if (self.shuffling):
                X, Y = self.unison_shuffle(X.T, Y.T)
                X = X.T
                Y = Y.T

            self.eta = self.weight_decay * self.eta

            for j in range(n_batch):
                j_start = j * self.batch_size
                j_end = (j + 1) * self.batch_size
                if j == n_batch - 1:
                    j_end = n

                Xbatch = X[:, j_start:j_end]
                Ybatch = Y[:, j_start:j_end]

                Pbatch = self.evaluateClassifier(Xbatch, self.W, self.b)
                if (self.SVM):
                    grad_W, grad_b = self.compute_gradients_SVM(
                        Xbatch, Ybatch, self.W, self.b)
                else:
                    grad_W, grad_b = self.compute_gradients(
                        Xbatch.T, Ybatch.T, Pbatch, self.W)

                self.W -= self.eta * grad_W
                self.b -= self.eta * grad_b

            val = self.computeCost(
                validationSet["data"], validationSet["one_hot"], self.W, self.b)[0]
            print("Validation loss: " + str(val))

            if val < bestVal:
                bestVal = np.copy(val)
                bestW = np.copy(self.W)
                bestb = np.copy(self.b)
                bestEpoch = np.copy(i)
                # print("New best: " + str(bestVal) + " epoch " + str(i))

            if (graphics):
                costsTraining.append(self.computeCost(X, Y, self.W, self.b)[0])
                costsValidation.append(val)

        self.W = np.copy(bestW)
        self.b = np.copy(bestb)
        print("Best epoch: " + str(bestEpoch))
        print("Best cost: " + str(self.computeCost(
            validationSet["data"], validationSet["one_hot"], self.W, self.b)[0]))

        if (graphics):
            c1 = plt.plot(costsTraining, label="Training cost")
            c2 = plt.plot(costsValidation, label="Validation cost")

            plt.xlabel('Epoch number')
            plt.ylabel('Cost')
            plt.title('Cost for the training and validation set over the epochs')
            plt.legend(loc='best')
            plt.savefig("training_validation_cost.png")
            plt.show()

            for i, row in enumerate(self.W):
                img = (row - row.min()) / (row.max() - row.min())
                plt.subplot(2, 5, i + 1)
                img = np.rot90(np.reshape(img, (32, 32, 3), order='F'), k=3)
                plt.imshow(img)
                plt.axis('off')
                plt.title(constants.labels[i])
            plt.savefig("weights.png")
            plt.show()

    def grid_search(self, X, Y, validationSet, etas, lambdas, weight_decays):
        bestW = np.copy(self.W)
        bestb = np.copy(self.b)
        bestEta = np.copy(self.eta)
        bestLambda = np.copy(self.lambda_reg)
        bestCost = self.computeCost(
            validationSet["data"], validationSet["one_hot"], self.W, self.b)[0]
        bestWeightDecay = np.copy(self.weight_decay)

        for eta in etas:
            for lambda_reg in lambdas:
                for weight_decay in weight_decays:
                    temp_eta = eta
                    print("START: eta: " + str(temp_eta) + " lambda_reg: " +
                          str(lambda_reg) + " weight_decay: " + str(weight_decay))

                    self.initialization()
                    self.eta = eta
                    self.lambda_reg = lambda_reg
                    self.weight_decay = weight_decay
                    self.fit(X, Y, validationSet=validationSet)
                    cost = self.computeCost(
                        validationSet["data"], validationSet["one_hot"], self.W, self.b)[0]

                    print(" eta: " + str(temp_eta) + " lambda_reg: " + str(lambda_reg) +
                          " weight_decay: " + str(weight_decay) + " cost: " + str(cost))

                    if cost < bestCost:
                        bestW = np.copy(self.W)
                        bestb = np.copy(self.b)
                        bestEta = temp_eta
                        bestLambda = np.copy(self.lambda_reg)
                        bestCost = cost
                        bestWeightDecay = weight_decay
                        print(" Best!")

        print("FINAL BEST => eta: " + str(bestEta) + " lambda_reg: " + str(bestLambda) +
              " weight_decay: " + str(weight_decay) + " cost: " + str(bestCost))
        self.W = np.copy(bestW)
        self.b = np.copy(bestb)
