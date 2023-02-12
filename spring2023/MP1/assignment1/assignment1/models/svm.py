"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
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
        self.n_samples = None
        self.n_features = None

    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        return

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.n_samples, self.n_features = X_train.shape
        self.w = np.random.rand(self.n_features, self.n_class)
        # self.w = np.load("svm_fashion.npy")
        ind_list = list(range(self.n_samples))
        np.random.shuffle(ind_list)
        X_train  = X_train[ind_list, :]
        y_train = y_train[ind_list,]
        self.X_train = X_train
        self.y_train = y_train
        for epoch in range(self.epochs):
            # for each batch
            for index in range(0, self.n_samples, self.batch_size):
                batch_X = self.X_train[index:min(index+self.batch_size, self.n_samples),:]
                batch_y = self.y_train[index:min(index+self.batch_size, self.n_samples)]
                batch_y = np.reshape(batch_y, (-1, 1))

                w_grad = np.zeros(self.w.shape)
                for i, xi in enumerate(batch_X):
                    wDotX = xi @ self.w # (1, n_class)
                    # print(xi.shape)
                    yi = batch_y[i]
                    updateClasses = (wDotX - wDotX[yi]) > -1  # 1 0 0 1 1 0 0 1... (1, n_class)
                    updateClasses[yi] = 0
                    numOfUpdate = np.sum(updateClasses)
                    updateGradient = xi.reshape((-1, 1)) * updateClasses
                    updateGradient[:, yi] = -xi.reshape((-1, 1)) * numOfUpdate
                    w_grad += updateGradient
                w_grad = w_grad / batch_X.shape[0]
                self.w = self.w - self.lr * w_grad
                self.w = (1-self.lr*self.reg_const/batch_X.shape[0]) * self.w
            decayRate = 0.99
            if(self.lr > 1e-5):
                self.lr *= 1/(1+epoch*decayRate)
            if(epoch % 10 == 0):
                pred = self.predict(self.X_train)
                acc = self.get_acc(pred, self.y_train)
                print(f"Epoch{epoch+1}, acc:{acc}")

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
        z = np.dot(X_test, self.w)
        return np.argmax(z, axis=1)
