"""Logistic regression model."""

import numpy as np

class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.batch_size = 30
        self.n_samples = None
        self.n_features = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-z))
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.n_samples, self.n_features = X_train.shape
        self.w = np.random.rand(self.n_features, 1)
        self.X_train = X_train
        self.y_train = y_train
        for epoch in range(self.epochs):
            # for each batch
            for index in range(0, self.n_samples, self.batch_size):
                batch_X = self.X_train[index:min(index+self.batch_size, self.n_samples),:]
                batch_y = self.y_train[index:min(index+self.batch_size, self.n_samples)]
                batch_y = np.reshape(batch_y, (-1, 1))

                # print((batch_y - self.sigmoid(np.dot(batch_X, self.w))))
                # print(batch_X)
                # print(-(batch_y - self.sigmoid(np.dot(batch_X, self.w)))*batch_X)
                w_grad = np.sum(-(batch_y - self.sigmoid(np.dot(batch_X, self.w)))*batch_X, axis=0)
                w_grad = np.reshape(w_grad, (-1, 1))
                self.w = self.w - self.lr * w_grad
            if(epoch % 10 == 0):
                pred_lr = self.predict(self.X_train)
                acc = self.get_acc(pred_lr, self.y_train)
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
        n_testSample = X_test.shape[0]
        z = np.dot(X_test, self.w)
        output = self.sigmoid(z)
        pred_label = (output>self.threshold) * np.ones((n_testSample, 1))
        pred_label = pred_label[:, 0]
        return pred_label
    
if __name__ == '__main__':
    lr = Logistic(lr=0.01, epochs=10, threshold=1.0)
    z = np.array([2.5, 3.6,1.2])
    s = lr.sigmoid(z)
    print(type(s))

