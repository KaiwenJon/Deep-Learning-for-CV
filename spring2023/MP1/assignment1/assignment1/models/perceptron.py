"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        
        self.batch_size = 30
        self.n_samples = None
        self.n_features = None
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        if(self.n_class == 2):
            self.n_samples, self.n_features = X_train.shape
            self.w = np.random.rand(self.n_features, 1)
            self.X_train = X_train
            self.y_train = y_train.copy()
            self.y_train[self.y_train == 0] = -1
            for epoch in range(self.epochs):
            # for each batch
                for index in range(0, self.n_samples, self.batch_size):
                    batch_X = self.X_train[index:min(index+self.batch_size, self.n_samples),:]
                    batch_y = self.y_train[index:min(index+self.batch_size, self.n_samples)]
                    batch_y = np.reshape(batch_y, (-1, 1))

                    # print((batch_y - self.sigmoid(np.dot(batch_X, self.w))))
                    # print(batch_X)
                    # print(-(batch_y - self.sigmoid(np.dot(batch_X, self.w)))*batch_X)
                    # print(batch_y.shape, np.dot(batch_X, self.w).shape, self.sigmoid(np.dot(batch_X, self.w)).shape, ((self.sigmoid(np.dot(batch_X, self.w)))*batch_X).shape)
                    filter = np.ones((batch_X.shape[0], 1))
                    filter[(batch_y * (batch_X @ self.w)) > 0] = 0
                    w_grad = np.sum(-filter * (batch_y * batch_X), axis=0) / batch_X.shape[0]
                    w_grad = np.reshape(w_grad, (-1, 1))
                    self.w = self.w - self.lr * w_grad
                if(epoch % 10 == 0):
                    pred = self.predict(self.X_train)
                    pred[pred == 0] = -1
                    acc = self.get_acc(pred, self.y_train)
                    print(f"Epoch{epoch+1}, acc:{acc}")
        elif(self.n_class > 2):
            self.n_samples, self.n_features = X_train.shape
            self.w = np.random.rand(self.n_features, self.n_class)

            ind_list = list(range(self.n_samples))
            np.random.shuffle(ind_list)
            X_train  = X_train[ind_list, :]
            y_train = y_train[ind_list,]

            self.X_train = X_train
            self.y_train = y_train
            self.opt_sum_sq_grad = np.zeros(self.w.shape)
            self.opt_momentum = np.zeros(self.w.shape)
            self.opt_RMSprop = np.zeros(self.w.shape)
            eps = 1e-8
            b1 = 0.9
            b2 = 0.999
            for epoch in range(self.epochs):
            # for each batch
                dif = 0
                for index in range(0, self.n_samples, self.batch_size):
                    batch_X = self.X_train[index:min(index+self.batch_size, self.n_samples),:]
                    batch_y = self.y_train[index:min(index+self.batch_size, self.n_samples)]
                    batch_y = np.reshape(batch_y, (-1, 1))

                    w_grad = np.zeros(self.w.shape)
                    for i, xi in enumerate(batch_X):
                        wDotX = xi @ self.w # (1, n_class)
                        yi = batch_y[i]
                        updateClasses = wDotX > wDotX[yi] # 1 0 0 1 1 0 0 1... (1, n_class)
                        numOfUpdate = np.sum(updateClasses)
                        updateGradient = xi.reshape((-1, 1)) * updateClasses
                        updateGradient[:, yi] = -xi.reshape((-1, 1)) * numOfUpdate
                        w_grad += updateGradient

                    w_grad = w_grad / batch_X.shape[0]
                    # self.opt_sum_sq_grad += np.square(w_grad)
                    self.opt_momentum = (1-b1) * w_grad + self.opt_momentum * b1
                    self.opt_RMSprop = b2 * self.opt_RMSprop + (1-b2) * np.square(w_grad) 
                    m = self.opt_momentum/(1-b1**(epoch+1))
                    v = self.opt_RMSprop/(1-b2**(epoch+1))
                    # self.w = self.w - self.lr * m / (np.sqrt(v)+eps)
                    # dif += np.sum(self.lr*w_grad)
                    # dif += np.sum(self.lr * m / (np.sqrt(v)+eps))
                    self.w = self.w - self.lr * w_grad
                    
                    # self.w = self.w - self.lr * w_grad / np.sqrt(self.opt_sum_sq_grad + eps)
                    
                decayRate = 0.5
                if(self.lr > 1e-6):
                    self.lr *= 1/(1+epoch*decayRate)
                if(epoch % 10 == 0):
                    pred = self.predict(self.X_train)
                    acc = self.get_acc(pred, self.y_train)
                    print(f"Epoch{epoch+1}, acc:{acc}, learning rate:{self.lr}")
        pass

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
        if(self.n_class == 2):
            n_testSample = X_test.shape[0]
            z = X_test @ self.w
            pred_label = (z > 0) * np.ones((n_testSample, 1))
            pred_label = pred_label[:, 0]
            return pred_label # 1 and 0
        elif(self.n_class > 2):
            z = np.dot(X_test, self.w)
            return np.argmax(z, axis=1)
