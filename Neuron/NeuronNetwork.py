import numpy as np
from fonctions.activation_function import ActivationFunction, Softmax, Sigmoid
from fonctions.loss_function import LossFunction, ClassificationCrossEntropy, BinaryCrossEntropy
import pickle
import os


# CLIPPING_W = 1000

class Layer:

    def __init__(self, input_size, output_size, activation_function, layer_index):
        np.random.seed(50)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))
        self.layer_index =  layer_index
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
            dz = da
        else:
            dz = da * self.activation_function.derivative(self.z)

        self.dW = np.dot(self.inputs.T, dz) / self.inputs.shape[0]
        self.db = np.sum(dz, axis=0, keepdims=True) / self.inputs.shape[0]

        dz_prev = np.dot(dz, self.weights.T)
        return dz_prev
    
    def setSkip_activation_derivative(self, skip: bool):
        self.skip_activation_derivative = skip

    def __repr__(self):
        return f"Layer_{self.layer_index}(input_size={self.weights.shape[0]}, output_size(neuron_count)={self.weights.shape[1]}, activation_function={self.activation_function.__class__.__name__})"
    
class Model:
    def __init__(self):
        self.layers = []
        self.loss_function: LossFunction = None
        self.learning_rate: float = 0.01

        self.train_loss_lst = []
        self.val_loss_lst = []

        self.train_accuracy_lst = []
        self.val_accuracy_lst = []

    def __repr__(self):
        description = "\n".join([str(layer) for layer in self.layers])
        if not description:
            return "Model: No layers"
        return f"Model:{len(self.layers)} layers" + '\n' + description

    def add_layer(self, output_size: int, activation_function: callable, input_size: int = None):
        if not self.layers and input_size is None:
            raise ValueError("First layer needs input_size")
        input_dim = input_size if not self.layers else self.layers[-1].weights.shape[1]
        self.layers.append(Layer(input_dim, output_size, activation_function, len(self.layers)))

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
        if np.isnan(loss) or np.isinf(loss):
            raise ValueError("Loss is NaN or Inf, check your data and model configuration.")

        dA = self.loss_function.derivative(y_batch, y_pred)

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

        for layer in self.layers:
            layer.weights -= self.learning_rate * layer.dW
            layer.bias -= self.learning_rate * layer.db

        return loss

    def _check_simplified_softmax_crossentropy(self):
        if isinstance(self.loss_function, ClassificationCrossEntropy) and isinstance(self.layers[-1].activation_function, Softmax):
            self.layers[-1].setSkip_activation_derivative(True)

    def train(self, train_set: tuple, val_set: tuple, epochs=1000, verbose=True, batch_size=32, early_stopping=True):
        X_train, Y_train = train_set
        X_val, Y_val = val_set

        self._check_simplified_softmax_crossentropy()
        self.fit(X_train, Y_train, X_val, Y_val, epochs, verbose, batch_size, early_stopping)

    def predict(self, X):
        y_pred = self.forward(X)
        if isinstance(self.loss_function, ClassificationCrossEntropy) and isinstance(self.layers[-1].activation_function, Softmax):
            return np.argmax(y_pred, axis=1)
        elif isinstance(self.layers[-1].activation_function, Sigmoid):
            return (y_pred > 0.5).astype(int)
        return y_pred
    
    def evaluate(self, X, Y):
        y_pred = self.forward(X)
        loss = self.loss_function.forward(Y, y_pred)

        if isinstance(self.loss_function, ClassificationCrossEntropy) and isinstance(self.layers[-1].activation_function, Softmax):
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(Y, axis=1))
        elif isinstance(self.layers[-1].activation_function, Sigmoid):
            accuracy = np.mean((y_pred > 0.5).astype(int) == Y)
        else:
            accuracy = None

        return loss, accuracy

    def save(self, file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, Model):
            raise ValueError("Loaded object is not a Model instance.")
        return model
    
    def _early_stopping(self, patience=10, val_loss=None):
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return True
            return False
        
    def fit(self, X_train, Y_train, X_val, Y_val, epochs=1000, verbose=True, batch_size=32, early_stopping=True):
        m = X_train.shape[0]
        np.random.seed(21)
        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            Y_train_shuffled = Y_train[indices]

            for i in range(0, m, batch_size):
                X_train_batch = X_train_shuffled[i:i + batch_size]
                Y_train_batch = Y_train_shuffled[i:i + batch_size]

                self.train_on_batch(X_train_batch, Y_train_batch)

            if verbose and (epoch % 1 == 0 or epoch == epochs - 1):

                loss_train, accuracy_train = self.evaluate(X_train, Y_train)
                self.train_loss_lst.append(loss_train)
                self.train_accuracy_lst.append(accuracy_train)

                loss_val, accuracy_val = self.evaluate(X_val, Y_val)
                self.val_loss_lst.append(loss_val)
                self.val_accuracy_lst.append(accuracy_val)

                print(f"-- Epoch {epoch+1}/{epochs} -- Loss_train: {loss_train:.4f} - Loss_val: {loss_val:.4f} -- Accuracy_train: {accuracy_train:.4f} - Accuracy_val: {accuracy_val:.4f} --")
            
            if self._early_stopping(patience=10,val_loss=loss_val) and early_stopping:
                print("Early stopping triggered.")
                break
