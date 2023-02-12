"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

        self.batch_size = 200

    def stable_softmax(self, X: np.ndarray):
        maximum = np.max(X, axis=1).reshape(-1, 1)
        exps = np.exp(X - maximum)
        total = np.sum(exps, axis=1).reshape(-1, 1)
        z = exps/total
        return z
    def cross_entropy(self, X, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        p = self.stable_softmax(X)
        p += 1e-40
        # We use multidimensional array indexing to extract 
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss
    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        # print(y_train)
        # print(self.n_class)
        # self.cross_entropy(X_train, y_train)
        # X - w - z - softmax - cross_entropy - l, sum all l-> L
        # print(y_train.shape)
        # print(np.dot(X_train, self.w))
        dLdz = self.stable_softmax(np.dot(X_train, self.w)) - y_train
        # print(dLdz.shape)
        dLdw = np.dot(X_train.T, dLdz)
        # print(X_train.shape)
        # print(dLdw.shape)
        w_grad = self.reg_const * self.w + dLdw/X_train.shape[0]
        return w_grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
        self.n_samples, self.n_features = X_train.shape

        ind_list = list(range(self.n_samples))
        np.random.shuffle(ind_list)
        X_train  = X_train[ind_list, :]
        y_train = y_train[ind_list,]



        # turn y into one hot
        y_train_oneHot = np.zeros((self.n_samples, self.n_class))
        y_train_oneHot[np.arange(self.n_samples), y_train] = 1
        self.w = np.random.rand(self.n_features, self.n_class)
        # self.w = np.load("./softmax_weights.npy") + np.random.rand(self.n_features, self.n_class)

        self.opt_sum_sq_grad = np.zeros(self.w.shape)
        self.opt_momentum = np.zeros(self.w.shape)
        self.opt_RMSprop = np.zeros(self.w.shape)
        eps = 1e-8
        b1 = 0.9
        b2 = 0.999

        for epoch in range(self.epochs):
            for index in range(0, self.n_samples, self.batch_size):
                # if (index != 0):
                #     continue
                batch_X = X_train[index:min(index + self.batch_size, self.n_samples),:]
                batch_y_oneHot = y_train_oneHot[index:min(index + self.batch_size, self.n_samples), :]
                w_grad = self.calc_gradient(batch_X, batch_y_oneHot)
                # self.w = self.w - self.lr/np.sqrt(lr_w) * w_grad

                # self.opt_momentum = (1-b1) * w_grad + self.opt_momentum * b1
                # self.opt_RMSprop = b2 * self.opt_RMSprop + (1-b2) * np.square(w_grad) 
                # m = self.opt_momentum/(1-b1**(epoch+1))
                # v = self.opt_RMSprop/(1-b2**(epoch+1))
                # self.w = self.w - self.lr * m / (np.sqrt(v)+eps)
                self.w = self.w - self.lr * w_grad
            decayRate = 0.99
            if(self.lr > 1e-4):
                self.lr *= 1/(1+epoch*decayRate)
            if(epoch % 10 == 0):
                pred_sf = self.predict(X_train)
                acc = self.get_acc(pred_sf, y_train)
                loss = self.cross_entropy(np.dot(X_train, self.w), y_train)
                print(f"Epoch{epoch+1}\t, acc:{acc:.4f}, loss:{loss:.4f}")
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me

        # X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
        z = np.dot(X_test, self.w)
        return np.argmax(z, axis=1)