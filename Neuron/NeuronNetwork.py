import numpy as np

class Neuron:
    def __init__(self,
                 input_size: int,
                 activation_function: callable,
                 random_weight: bool = True
                 ):
        
        self.input_size = input_size
        self.activation_function = activation_function
        self.weights = np.random.rand(input_size) if random_weight else np.zeros(input_size)
        # print(f"weights: {self.weights}")
    
    def forward(self, inputs):
        if inputs.shape[1] != self.input_size:
            raise ValueError(f"Input size must be {self.input_size}, but got {inputs.shape[1]}")
        
        print(f"input{inputs}\nweights{self.weights.reshape(-1, 1)}")

        z = np.dot(inputs, self.weights.reshape(-1, 1))
        print(z)
        self.output = self.activation_function(z)
        return self.output
    
class Layer:
    def __init__(self,
                 neuron_count: int,
                 activation_function: callable,
                 input_size: int = None,
                 random_weight: bool = True
                 ):
        
        self.neuron_count = neuron_count
        self.input_size = input_size
        self.activation_function = activation_function
        self.random_weight = random_weight
        self.neurons = []

        if input_size is not None:
            self.neurons = [Neuron(input_size, activation_function, random_weight) for _ in range(neuron_count)]
    
    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
    
    def set_input_size(self, input_size: int):
        if self.input_size is None:
            self.input_size = input_size
            self.neurons = [Neuron(input_size=input_size, 
                                   activation_function=self.activation_function, 
                                   random_weight=self.random_weight) for _ in range(self.neuron_count)] 

class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self,
                  neuron_count: int,
                  activation_function: callable,
                  random_weight: bool = True,
                  input_size: int = None
                ):
        
        if len(self.layers) == 0 and input_size is None:
            raise ValueError("Input size must be specified for the first layer.")
        if len(self.layers) > 0 and input_size is not None:
            raise ValueError("Input size should not be specified for layers after the first one.")       
        
        if len(self.layers) == 0:
            layer = Layer(neuron_count, activation_function, input_size, random_weight)
        else:
            layer = Layer(neuron_count, activation_function, self.layers[-1].input_size, random_weight)
        self.layers.append(layer)
    
    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: input shape {inputs.shape}")
            inputs = layer.forward(inputs)
        return inputs