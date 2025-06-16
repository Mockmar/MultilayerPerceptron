import numpy as np
from fonctions.activation_function import ActivationFunction, Softmax
from fonctions.loss_function import LossFunction, ClassificationCrossEntropy

    
class Layer:

    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)  # He init (utile pour ReLU)
        self.bias = np.zeros((1, output_size))
        self.activation_function = activation_function
        self.skip_activation_derivative = False
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
        if self.skip_activation_derivative:
            dz = da  # simplification Softmax + CrossEntropy
        else:
            dz = da * self.activation_function.derivative(self.z)
        self.dW = np.dot(self.inputs.T, dz) / self.inputs.shape[0]  # moyenne sur batch
        self.db = np.sum(dz, axis=0, keepdims=True) / self.inputs.shape[0]
        dz_prev = np.dot(dz, self.weights.T)
        return dz_prev
    
    def setSkip_activation_derivative(self, skip: bool):
        self.skip_activation_derivative = skip

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

    def train_on_batch(self, X_batch, y_batch):
        y_pred = self.forward(X_batch)

        loss = self.loss_function.forward(y_batch, y_pred)

        if isinstance(self.loss_function, ClassificationCrossEntropy) and isinstance(self.layers[-1].activation_function, Softmax):
            dA = y_pred - y_batch
            self.layers[-1].skip_activation_derivative = True
        else:
            dA = self.loss_function.derivative(y_batch, y_pred)
            self.layers[-1].skip_activation_derivative = False

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

        for layer in self.layers:
            layer.weights -= self.learning_rate * layer.dW
            layer.bias -= self.learning_rate * layer.db

        return loss

    def _check_simplified_softmax_crossentropy(self):
        if isinstance(self.loss_function, ClassificationCrossEntropy) and isinstance(self.layers[-1].activation_function, Softmax):
            self.layers[-1].skip_activation_derivative = True


    def fit(self, X, y, epochs=1000, verbose=True, batch_size=32):
        m = X.shape[0]

        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.train_on_batch(X_batch, y_batch)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                y_pred_full = self.forward(X)
                loss_full = self.loss_function.forward(y, y_pred_full)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_full:.4f}")

