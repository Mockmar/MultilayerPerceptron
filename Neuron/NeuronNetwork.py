import numpy as np
from fonctions.activation_function import ActivationFunction
from fonctions.loss_function import LossFunction

    
class Layer:

    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.activation_function = activation_function
        self.inputs = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None

    def __repr__(self):
        return f"Layer(input_size={self.weights.shape[0]}, output_size(neuron_count)={self.weights.shape[1]}, activation_function={self.activation_function.__class__.__name__})"

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.a = self.activation_function.forward(self.z)
        return self.a

    def backward(self, dA, learning_rate, apply_activation=True):
        # dz = dA * self.activation_function.derivative(self.z) if apply_activation else dA
        dz = dA * self.activation_function.derivative(self.z)
        m = self.inputs.shape[0]
        self.dW = np.dot(self.inputs.T, dz) / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m
        dA_prev = np.dot(dz, self.weights.T)
        # self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db

        print("Poids avant update:", self.weights[0, :5])  # par ex., quelques poids
        self.weights -= learning_rate * self.dW
        print("Poids après update:", self.weights[0, :5])
        return dA_prev
    
class Model:
    def __init__(self):
        self.layers = []
        self.loss_function: LossFunction = None
        self.learning_rate: float = 0.01

    def __repr__(self):
        description = "\n".join([str(layer) for layer in self.layers])
        if not description:
            return "Model: No layers"
        return f"Model:{len(self.layers)} layers" + '\n' + description

    def add_layer(self, output_size: int, activation_function: callable, input_size: int = None):
        if not self.layers and input_size is None:
            raise ValueError("First layer needs input_size")
        input_dim = input_size if not self.layers else self.layers[-1].weights.shape[1]
        self.layers.append(Layer(input_dim, output_size, activation_function))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, y_pred, y_true):
        dA = self.loss_function.derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            dA = layer.backward(dA, self.learning_rate)

    def compile(self, loss_function: LossFunction, learning_rate: float = 0.01):
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=10):
        for epoch in range(epochs):
            y_pred = self.forward(X)

            loss = self.loss_function.forward(y_pred, y)

            self.backward(y_pred, y)

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return y_pred



