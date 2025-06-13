import numpy as np
from fonctions.activation_function import ActivationFunction
from fonctions.loss_function import LossFunction

    
class Layer:

    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)  # He init (utile pour ReLU)
        self.bias = np.zeros((1, output_size))
        self.activation_function = activation_function
        self.inputs = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.a = self.activation_function.forward(self.z)
        return self.a

    def backward(self, da):
        dz = da * self.activation_function.derivative(self.z)  # dérivée de z
        self.dW = np.dot(self.inputs.T, dz) / self.inputs.shape[0]  # moyenne sur batch
        self.db = np.sum(dz, axis=0, keepdims=True) / self.inputs.shape[0]
        dz_prev = np.dot(dz, self.weights.T)
        return dz_prev

    def __repr__(self):
        return f"Layer(input_size={self.weights.shape[0]}, output_size(neuron_count)={self.weights.shape[1]}, activation_function={self.activation_function.__class__.__name__})"
    
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

    def compile(self, loss_function: LossFunction, learning_rate: float = 0.01):
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            # Forward
            y_pred = self.forward(X)

            # Calcul de la perte
            loss = self.loss_function.forward(y, y_pred)

            # Backward : on commence par la dérivée de la perte
            dA = self.loss_function.derivative(y, y_pred)

            # Rétropropagation à travers les couches (ordre inverse)
            for layer in reversed(self.layers):
                dA = layer.backward(dA)

            # Mise à jour des poids
            for layer in self.layers:
                layer.weights -= self.learning_rate * layer.dW
                layer.bias -= self.learning_rate * layer.db

            # Affichage optionnel
            if verbose and (epoch % 10000 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
