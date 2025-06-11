import numpy as np

    
class Layer:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation_function: callable
                 ):
        
        # output_size is the number of neurons in the layer
        # input_size is the number of features in the input data
        # les biais sont dans les poids
        self.weights = np.random.randn(input_size, output_size)
        self.activation_function = activation_function

    def forward(self, inputs):
        self.inputs = inputs
        z = np.dot(inputs, self.weights)
        return self.activation_function(z)

class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, output_size: int, activation_function: callable, input_size: int = None):
        if not self.layers and input_size is None:
            raise ValueError("First layer needs input_size")
        input_dim = input_size if not self.layers else self.layers[-1].weights.shape[1]
        self.layers.append(Layer(input_dim, output_size, activation_function))

    def forward(self, X):
        for i, layer in enumerate(self.layers):
            X = layer.forward(X)
        return X
    
