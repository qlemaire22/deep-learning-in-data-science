import numpy as np
import constants
from math import floor, sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt


class Classifier():
    def __init__(self, learning_rate=0.015, regularization_term=0.00035, batch_size=100, n_epochs=50, weight_decay=0.9, shuffling=False, hidden_nodes=200, rho=0.9, leaky_RELU=True):
        self.W2 = np.zeros((constants.K, hidden_nodes))
        self.W1 = np.zeros((hidden_nodes, constants.d))
        self.b2 = np.zeros((constants.K, 1))
        self.b1 = np.zeros((hidden_nodes, 1))
        self.W2_momentum = np.zeros((constants.K, hidden_nodes))
        self.W1_momentum = np.zeros((hidden_nodes, constants.d))
        self.b2_momentum = np.zeros((constants.K, 1))
        self.b1_momentum = np.zeros((hidden_nodes, 1))

        self.eta = learning_rate
        self.lambda_reg = regularization_term
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.shuffling = shuffling
        self.hidden_nodes = hidden_nodes
        self.rho = rho

        self.leaky_RELU = leaky_RELU

        np.random.seed(1)

        self.initialization()

    def initialization(self):
        mu = 0
        sigma = sqrt(2) / sqrt(constants.d)

        self.W1 = np.random.normal(mu, sigma, self.W1.shape)
        self.W2 = np.random.normal(mu, sigma, self.W2.shape)
        self.b2 = np.zeros((constants.K, 1))
        self.b1 = np.zeros((self.hidden_nodes, 1))

        self.W2_momentum = np.zeros((constants.K, self.hidden_nodes))
        self.W1_momentum = np.zeros((self.hidden_nodes, constants.d))
        self.b2_momentum = np.zeros((constants.K, 1))
        self.b1_momentum = np.zeros((self.hidden_nodes, 1))

    def evaluateClassifier(self, X, W1, b1, W2, b2):
        s1 = np.dot(W1, X) + b1
        if (self.leaky_RELU):
            h = np.maximum(s1, 0.01 * s1)
        else:
            h = np.maximum(s1, 0)
        s2 = np.dot(W2, h) + b2
        P = self.softmax(s2)
        assert(P.shape == (constants.K, X.shape[1]))
        return P

    def softmax(self, x):
        r = np.exp(x) / sum(np.exp(x))
        return r

    def computeCost(self, X, Y, W1, b1, W2, b2):
        regularization = self.lambda_reg * \
            (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        loss_sum = 0
        for i in range(X.shape[1]):
            x = np.zeros((constants.d, 1))
            y = np.zeros((constants.K, 1))
            x = X[:, [i]]
            y = Y[:, [i]]
            loss_sum += self.cross_entropy(x, y, W1, b1, W2, b2)
        loss_sum /= X.shape[1]
        final = loss_sum + regularization
        assert(len(final) == 1)
        return final

    def cross_entropy(self, x, y, W1, b1, W2, b2):
        l = - np.log(np.dot(y.T, self.evaluateClassifier(x, W1, b1, W2, b2)))
        assert(len(l) == 1)
        return l

    def computeAccuracy(self, X, Y):
        acc = 0
        for i in range(X.shape[1]):
            P = self.evaluateClassifier(
                X[:, [i]], self.W1, self.b1, self.W2, self.b2)
            label = np.argmax(P)
            if label == Y[i]:
                acc += 1
        acc /= X.shape[1]
        return acc

    def compute_gradients(self, X, Y, P, W1, W2, b1):
        S1 = np.dot(W1, X) + b1

        if (self.leaky_RELU):
            H = np.maximum(S1, 0.1 * S1)
        else:
            H = np.maximum(S1, 0)

        G = -(Y.T - P.T).T

        gradb2 = np.mean(G, axis=-1, keepdims=True)
        gradW2 = np.dot(G, H.T)
        G = np.dot(G.T, W2)
        if (self.leaky_RELU):
            S1 = np.where(S1 > 0, 1, 0.01)
        else:
            S1 = np.where(S1 > 0, 1, 0)
        G = np.multiply(G.T, S1)
        gradb1 = np.mean(G, axis=-1, keepdims=True)
        gradW1 = np.dot(G, X.T)

        n = X.shape[1]

        gradW1 /= n
        gradW2 /= n
        gradW1 += 2 * self.lambda_reg * W1
        gradW2 += 2 * self.lambda_reg * W2

        self.W1_momentum = self.W1_momentum * self.rho + self.eta * gradW1
        self.W2_momentum = self.W2_momentum * self.rho + self.eta * gradW2
        self.b1_momentum = self.b1_momentum * self.rho + self.eta * gradb1
        self.b2_momentum = self.b2_momentum * self.rho + self.eta * gradb2

    def unison_shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def fit(self, X, Y, validationSet=[], graphics=False):
        n = X.shape[1]
        costsTraining = []
        costsValidation = []
        bestW1 = np.copy(self.W1)
        bestb1 = np.copy(self.b1)
        bestW2 = np.copy(self.W2)
        bestb2 = np.copy(self.b2)
        bestVal = self.computeCost(
            validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[0]
        bestEpoch = 0

        for i in tqdm(range(self.n_epochs)):
            n_batch = floor(n / self.batch_size)

            if (self.shuffling):
                X, Y = self.unison_shuffle(X.T, Y.T)
                X = X.T
                Y = Y.T

            #self.eta = self.weight_decay * self.eta
            if i%10 == 0:
                if i != 0:
                    self.eta = 0.1 * self.eta

            for j in range(n_batch):
                j_start = j * self.batch_size
                j_end = (j + 1) * self.batch_size
                if j == n_batch - 1:
                    j_end = n

                Xbatch = X[:, j_start:j_end]
                Ybatch = Y[:, j_start:j_end]

                Pbatch = self.evaluateClassifier(
                    Xbatch, self.W1, self.b1, self.W2, self.b2)

                self.compute_gradients(
                    Xbatch, Ybatch, Pbatch, self.W1, self.W2, self.b1)

                self.W1 -= self.W1_momentum
                self.b1 -= self.b1_momentum
                self.W2 -= self.W2_momentum
                self.b2 -= self.b2_momentum

            val = self.computeCost(
                validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[0]
            print(val)
            if val < bestVal:
                bestVal = np.copy(val)
                bestW1 = np.copy(self.W1)
                bestb1 = np.copy(self.b1)
                bestW2 = np.copy(self.W2)
                bestb2 = np.copy(self.b2)
                bestEpoch = np.copy(i)
                # print("New best: " + str(bestVal) + " epoch " + str(i))

            if (graphics):
                costsTraining.append(self.computeCost(
                    X, Y, self.W1, self.b1, self.W2, self.b2)[0])
                costsValidation.append(val)
        print("Final loss: " + str(val))
        #
        self.W1 = np.copy(bestW1)
        self.b1 = np.copy(bestb1)
        self.W2 = np.copy(bestW2)
        self.b2 = np.copy(bestb2)

        print("Best epoch: " + str(bestEpoch))
        print("Best cost: " + str(self.computeCost(
            validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[0]))

        if (graphics):
            c1 = plt.plot(costsTraining, label="Training cost")
            c2 = plt.plot(costsValidation, label="Validation cost")

            plt.xlabel('Epoch number')
            plt.ylabel('Cost')
            plt.title('Cost for the training and validation set over the epochs')
            plt.legend(loc='best')
            plt.savefig("training_validation_cost.png")
            plt.show()


    def grid_search(self, X, Y, validationSet, etas, lambdas, weight_decays):
        bestW1 = np.copy(self.W1)
        bestb1 = np.copy(self.b1)
        bestW2 = np.copy(self.W2)
        bestb2 = np.copy(self.b2)
        bestEta = np.copy(self.eta)
        bestLambda = np.copy(self.lambda_reg)
        bestWeightDecay = np.copy(self.weight_decay)

        bestCost = self.computeCost(
            validationSet["data"], validationSet["one_hot"],self.W1, self.b1, self.W2, self.b2)[0]

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
                        validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[0]

                    print(" eta: " + str(temp_eta) + " lambda_reg: " + str(lambda_reg) +
                          " weight_decay: " + str(weight_decay) + " cost: " + str(cost))

                    if cost < bestCost:
                        bestW1 = np.copy(self.W1)
                        bestb1 = np.copy(self.b1)
                        bestW2 = np.copy(self.W2)
                        bestb2 = np.copy(self.b2)
                        bestEta = temp_eta
                        bestLambda = np.copy(self.lambda_reg)
                        bestCost = cost
                        bestWeightDecay = weight_decay
                        print(" Best!")

        print("FINAL BEST => eta: " + str(bestEta) + " lambda_reg: " + str(bestLambda) +
              " weight_decay: " + str(weight_decay) + " cost: " + str(bestCost))
        self.W1 = np.copy(bestW1)
        self.b1 = np.copy(bestb1)
        self.W2 = np.copy(bestW2)
        self.b2 = np.copy(bestb2)

    def random_search(self, X, Y, validationSet, search_number, etas, lambdas):
        bestW1 = np.copy(self.W1)
        bestb1 = np.copy(self.b1)
        bestW2 = np.copy(self.W2)
        bestb2 = np.copy(self.b2)
        bestEta = np.copy(self.eta)
        bestLambda = np.copy(self.lambda_reg)

        bestCost = self.computeCost(
            validationSet["data"], validationSet["one_hot"],self.W1, self.b1, self.W2, self.b2)[0]

        for i in range(search_number):
            eta = np.random.uniform(etas[0], etas[1])
            lambda_reg = np.random.uniform(lambdas[0], lambdas[1])
            temp_eta = eta

            print("START: eta: " + str(temp_eta) + " lambda_reg: " +
                  str(lambda_reg))

            self.initialization()
            self.eta = eta
            self.lambda_reg = lambda_reg

            self.fit(X, Y, validationSet=validationSet)

            cost = self.computeCost(
                validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[0]

            print(" eta: " + str(temp_eta) + " lambda_reg: " + str(lambda_reg) + " cost: " + str(cost))

            if cost < bestCost:
                bestW1 = np.copy(self.W1)
                bestb1 = np.copy(self.b1)
                bestW2 = np.copy(self.W2)
                bestb2 = np.copy(self.b2)
                bestEta = temp_eta
                bestLambda = np.copy(self.lambda_reg)
                bestCost = cost
                print(" Best!")

        print("FINAL BEST => eta: " + str(bestEta) + " lambda_reg: " + str(bestLambda) + " cost: " + str(bestCost))
        self.W1 = np.copy(bestW1)
        self.b1 = np.copy(bestb1)
        self.W2 = np.copy(bestW2)
        self.b2 = np.copy(bestb2)

    #### For testing ####

    def computeGradientsNum(self, X, Y, W1, b1, W2, b2):
        h = 1e-5
        n = X.shape[1]

        grad_W2 = np.zeros((constants.K, self.hidden_nodes))
        grad_W1 = np.zeros((self.hidden_nodes, constants.d))
        grad_b2 = np.zeros((constants.K, 1))
        grad_b1 = np.zeros((self.hidden_nodes, 1))

        c = self.computeCost(X, Y, W1, b1, W2, b2)

        print("Computing b gradient")

        for i in range(b1.shape[0]):
            b_try_1 = np.copy(b1)
            b_try_1[i] += h
            c2 = self.computeCost(X, Y, W1, b_try_1, W2, b2)
            grad_b1[i] = (c2 - c) / h

        for i in range(b2.shape[0]):
            b_try_2 = np.copy(b2)
            b_try_2[i] += h
            c2 = self.computeCost(X, Y, W1, b1, W2, b_try_2)
            grad_b2[i] = (c2 - c) / h

        print("Computing W1 gradient")

        for i in tqdm(range(W1.shape[0])):
            for j in range(W1.shape[1]):
                W_try_1 = np.copy(W1)
                W_try_1[i][j] += h
                c2 = self.computeCost(X, Y, W_try_1, b1, W2, b2)
                grad_W1[i][j] = (c2 - c) / h

        print("Computing W2 gradient")

        for i in tqdm(range(W2.shape[0])):
            for j in range(W2.shape[1]):
                W_try_2 = np.copy(W2)
                W_try_2[i][j] += h
                c2 = self.computeCost(X, Y, W1, b1, W_try_2, b2)
                grad_W2[i][j] = (c2 - c) / h

        return grad_W1, grad_W2, grad_b1, grad_b2

    def computeGradientsNumSlow(self, X, Y, W1, b1, W2, b2):
        h = 1e-5
        n = X.shape[1]

        grad_W2 = np.zeros((constants.K, self.hidden_nodes))
        grad_W1 = np.zeros((self.hidden_nodes, constants.d))
        grad_b2 = np.zeros((constants.K, 1))
        grad_b1 = np.zeros((self.hidden_nodes, 1))

        c = self.computeCost(X, Y, W1, b1, W2, b2)

        print("Computing b gradient")

        for i in range(b1.shape[0]):
            b_try_1 = np.copy(b1)
            b_try_1[i] -= h
            c1 = self.computeCost(X, Y, W1, b_try_1, W2, b2)
            b_try_1 = np.copy(b1)
            b_try_1[i] += h
            c2 = self.computeCost(X, Y, W1, b_try_1, W2, b2)
            grad_b1[i] = (c2 - c1) / (2 * h)

        for i in range(b2.shape[0]):
            b_try_2 = np.copy(b2)
            b_try_2[i] -= h
            c1 = self.computeCost(X, Y, W1, b1, W2, b_try_2)
            b_try_2 = np.copy(b2)
            b_try_2[i] += h
            c2 = self.computeCost(X, Y, W1, b1, W2, b_try_2)
            grad_b2[i] = (c2 - c1) / (2 * h)

        print("Computing W1 gradient")

        for i in tqdm(range(W1.shape[0])):
            for j in range(W1.shape[1]):
                W_try_1 = np.copy(W1)
                W_try_1[i][j] -= h
                c1 = self.computeCost(X, Y, W_try_1, b1, W2, b2)
                grad_W1[i][j] = (c2 - c1) / (2 * h)
                W_try_1 = np.copy(W1)
                W_try_1[i][j] += h
                c2 = self.computeCost(X, Y, W_try_1, b1, W2, b2)
                grad_W1[i][j] = (c2 - c1) / (2 * h)

        print("Computing W2 gradient")

        for i in tqdm(range(W2.shape[0])):
            for j in range(W2.shape[1]):
                W_try_2 = np.copy(W2)
                W_try_2[i][j] -= h
                c1 = self.computeCost(X, Y, W1, b1, W_try_2, b2)
                grad_W2[i][j] = (c2 - c1) / (2 * h)
                W_try_2 = np.copy(W2)
                W_try_2[i][j] += h
                c2 = self.computeCost(X, Y, W1, b1, W_try_2, b2)
                grad_W2[i][j] = (c2 - c1) / (2 * h)

        return grad_W1, grad_W2, grad_b1, grad_b2
