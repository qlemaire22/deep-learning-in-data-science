import matplotlib.pyplot as plt
import numpy as np

def plot():
    with open("loss.npz.npy") as f:
        loss = list(np.load("loss.npz.npy").reshape(3102, 1))
    print(loss)
    loss_plot = plt.plot(loss, label="training loss")
    plt.xlabel('epoch (divided by 100)')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('graph.png')
    plt.show()


if __name__ == '__main__':
    plot()
