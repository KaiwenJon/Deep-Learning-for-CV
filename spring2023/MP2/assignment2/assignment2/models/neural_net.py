"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        optimizer: str
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.optimizer = optimizer

        self.opt_momentum = {}
        self.opt_sum_sq_grad = {}
        self.opt_RMSprop = {}

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            # self.params["W" + str(i)] = np.random.rand(sizes[i - 1], sizes[i])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            self.opt_momentum["W" + str(i)] = np.zeros((sizes[i-1], sizes[i]))
            self.opt_momentum["b" + str(i)] = np.zeros(sizes[i])
            self.opt_sum_sq_grad["W" + str(i)] = np.zeros((sizes[i-1], sizes[i]))
            self.opt_sum_sq_grad["b" + str(i)] = np.zeros(sizes[i])
            self.opt_RMSprop["W" + str(i)] = np.zeros((sizes[i-1], sizes[i]))
            self.opt_RMSprop["b" + str(i)] = np.zeros(sizes[i])
        # for key, value in self.params.items() :
        #     print(key)
    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        # X: (n_samples, input_D)
        # W: (input_D, output_D)
        # b: (, output_D)
        # np.dot(X, W) gives us (n_samples, outputD)
        # for each row(sample), we add b
        # output: (n_samples, output_D)
        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X) # element-wise

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        # X: (n_samples, n_class)
        # For every row, do softmax itself.
        maximum = np.max(X, axis=1).reshape(-1, 1)
        exps = np.exp(X - maximum)
        total = np.sum(exps, axis=1).reshape(-1, 1)
        y = exps/total
        # print("X\n", X)
        # print("y\n", y)
        # print("maximum\n", maximum)
        # print("softmax\n", y)
        return y

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return 1 / (1 + np.exp(-x))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return np.sum((y - p)**2) / y.shape[0]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        self.outputs = {}
        self.outputs["X0"] = X # X0: X_train (n_samples, inputD)
        self.outputs["A0"] = X # No activation
        for i in range(1, self.num_layers + 1):
            # print(i) # 1, 2, ..., num_layers
            w = self.params["W" + str(i)]
            x = self.outputs["A" + str(i-1)]
            b = self.params["b" + str(i)]
            Z = self.linear(w, x, b) # (n_samples, output_D)
            # print(i, np.max(Z))
            # print(Z[0:5, 0:8])
            self.outputs["X" + str(i)] = Z
            if(i != self.num_layers):
                # print("relu", i)
                A = self.relu(Z) # (n_samples, output_D)
            else:
                # print("softmax", i)
                A = self.sigmoid(Z) # (n_samples, n_class)
            self.outputs["A" + str(i)] = A
        # score = self.outputs["A" + str(self.num_layers)] # last layer, after sigmoid
        # Within self.outputs,
        # X1, X2, ... X(num_layers) represent neuron before activation
        # A1, A2, ... A(num_layers) represent neuron after activation
        # print("--------Output------------")
        # for key, value in self.outputs.items() :
        #     print(key, "\n", value)
        return self.outputs["A" + str(self.num_layers)]
    def cross_entropy(self, y_output, y_train):
        y_output += 1e-40
        log_likelihood = -np.log(y_output[range(len(y_train)), y_train])
        # print("y_output\n", y_output)
        # print("log of y_output\n",np.log(y_output))
        # y_train_oneHot = np.zeros((self.n_samples, self.n_class))
        # y_train_oneHot[np.arange(self.n_samples), y_train] = 1
        # print("y_train\n",y_train_oneHot)
        # print("y_output[range(self.n_samples), y_train]\n",y_output[range(self.n_samples), y_train])
        # print("log_likelihood\n",log_likelihood)
        # print("Myway:\n",np.multiply(np.log(y_output), y_train_oneHot))
        loss = np.sum(log_likelihood)
        return loss

    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        n_samples = y.shape[0]
        self.gradients = {}
        # last layer (dL/dZ)
        # for softmax & cross-entroyp => self.gradients["X" + str(self.num_layers)] = self.outputs["A" + str(self.num_layers)] - y_train_oneHot #(n_samples, n_class)
        sig_output = self.outputs["A" + str(self.num_layers)]
        self.gradients["X" + str(self.num_layers)] = 1/n_samples * 2 * (sig_output - y) * (sig_output * (np.ones(sig_output.shape) - sig_output))
        for i in range(self.num_layers, 0, -1):
            # print(i) # num_layers, ..., 2, 1 ex: i=3
            grad_w = self.outputs["A" + str(i-1)].T @ self.gradients["X" + str(i)] # ex: dL/dw3 = a2.T @ dL/dx3, where a2 is neuron after relu
            grad_a = self.gradients["X" + str(i)] @ self.params["W" + str(i)].T # ex: dL/da2 = dl/dx3 @ w3.T, 
            grad_x = np.multiply(grad_a, self.outputs["X" + str(i-1)] > 0) # element-wise, dL/dx2 = dL/da2 * (whether x2 > 0), gradient of relu
            grad_b = np.sum(self.gradients["X" + str(i)], axis = 0) # dL/db3 = sum(dL/dx3, axis=0)
            self.gradients["W" + str(i)] = grad_w  # dL/dw3
            self.gradients["X" + str(i-1)] = grad_x # dL/dx2
            self.gradients["b" + str(i)] = grad_b # dL/db3

            # print("grad_w\n", grad_w[0:5, 0:4])
            # print("grad_b\n", grad_b[0:5])
        # print("==================gradient================")
        # for key, value in self.gradients.items() :
        #     print(key, "\n", value)
        # calculate cross entropy between self.outputs[A$(num_layers)] and y (not one hot)
        
        # np.set_printoptions(precision=2)
        # print("y_output\n",y_output)
        loss = self.mse(y=y, p=sig_output)
        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-15,
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.

        opt = self.optimizer
        for i in range(self.num_layers, 0, -1):
            # print(i) # num_layers, ..., 2, 1 ex: i=3
            para_name_w = "W" + str(i)
            para_name_b = "b" + str(i)
            if(opt == "SGD"):
                self.params[para_name_w] -= lr * self.gradients[para_name_w]
                self.params[para_name_b] -= lr * self.gradients[para_name_b]
            elif(opt == "SGDM"):
                self.opt_momentum[para_name_w] = -lr * self.gradients[para_name_w] + self.opt_momentum[para_name_w]* b1
                self.opt_momentum[para_name_b] = -lr * self.gradients[para_name_b] + self.opt_momentum[para_name_b]* b1
                self.params[para_name_w] += self.opt_momentum[para_name_w]
                self.params[para_name_b] += self.opt_momentum[para_name_b]
            elif(opt == "Adagrad"):
                self.opt_sum_sq_grad[para_name_w] += np.square(self.gradients[para_name_w])
                self.opt_sum_sq_grad[para_name_b] += np.square(self.gradients[para_name_b])
                self.params[para_name_w] -= lr * self.gradients[para_name_w] / (np.sqrt(self.opt_sum_sq_grad[para_name_w])+eps)
                self.params[para_name_b] -= lr * self.gradients[para_name_b] / (np.sqrt(self.opt_sum_sq_grad[para_name_b])+eps)
                # print(para_name_w, "\n", self.params[para_name_w][0:3][0:3])
            elif(opt == "RMSProp"):
                self.opt_RMSprop[para_name_w]  = b2 * self.opt_RMSprop[para_name_w] + (1 - b2) * np.square(self.gradients[para_name_w])
                self.opt_RMSprop[para_name_b]  = b2 * self.opt_RMSprop[para_name_b] + (1 - b2) * np.square(self.gradients[para_name_b])
                self.params[para_name_w] -= lr * self.gradients[para_name_w] / (np.sqrt(self.opt_RMSprop[para_name_w])+eps)
                self.params[para_name_b] -= lr * self.gradients[para_name_b] / (np.sqrt(self.opt_RMSprop[para_name_b])+eps)
            elif(opt == "Adam"):
                self.opt_momentum[para_name_w] = (1-b1) * self.gradients[para_name_w] + self.opt_momentum[para_name_w]* b1
                self.opt_momentum[para_name_b] = (1-b1) * self.gradients[para_name_b] + self.opt_momentum[para_name_b]* b1
                self.opt_RMSprop[para_name_w]  = b2 * self.opt_RMSprop[para_name_w] + (1 - b2) * np.square(self.gradients[para_name_w])
                self.opt_RMSprop[para_name_b]  = b2 * self.opt_RMSprop[para_name_b] + (1 - b2) * np.square(self.gradients[para_name_b])
                self.params[para_name_w] -= lr * self.opt_momentum[para_name_w] / (np.sqrt(self.opt_RMSprop[para_name_w])+eps)
                self.params[para_name_b] -= lr * self.opt_momentum[para_name_b] / (np.sqrt(self.opt_RMSprop[para_name_b])+eps)
            
                
                
        # print("***********params***********")
        # for key, value in self.params.items() :
        #     print(key, "\n", value)
