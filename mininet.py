#!/usr/bin/env python

import numpy as np

from minigrad.tensor import Tensor
from minigrad.utils import layer_init_uniform, fetch_mnist
import minigrad.optimizer as optim
from tqdm import trange

np.random.seed(44)

# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()

# create a model
class MiniNet:
    def __init__(self):
        self.l1 = Tensor(layer_init_uniform(784, 200))
        self.l2 = Tensor(layer_init_uniform(200, 25))
        self.l3 = Tensor(layer_init_uniform(25, 10))

    def forward(self, x):
        x = x.dot(self.l1).relu()
        x = x.dot(self.l2).relu()
        x = x.dot(self.l3).logsoftmax()
        return x

model = MiniNet()
optim = optim.SGD([model.l1, model.l2, model.l3], lr=1e-2)

batch_size = 128
losses, accuracies = [], []

for i in (t:= trange(3000)):
    sample = np.random.randint(0, X_train.shape[0], size=(batch_size))

    x = Tensor(X_train[sample].reshape(-1, 28*28))
    Y = Y_train[sample]
    y = np.zeros((len(sample),10), np.float32)
    y[range(y.shape[0]), Y] = -10.0
    y = Tensor(y)

    # network
    out = model.forward(x)

    # Loss Function
    loss = out.mul(y).mean()
    loss.backward()
    optim.step()


    cat = np.argmax(out.data, axis=1)
    accuracy = (cat == Y).mean()

    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.5f accuracy %.5f" % (loss, accuracy))

def eval():
    Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()


accuracy = eval()
print("Test set Accuracy : %.5f" % accuracy)


