import numpy as np

class ActivationFunction:
    
    @staticmethod
    def forward(x):
        raise NotImplementedError("Forward method not implemented.")
    
    @staticmethod
    def derivative(x):
        raise NotImplementedError("Derivative method not implemented.")
    

class Sigmoid(ActivationFunction):
    
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        s = Sigmoid.forward(x)
        return s * (1 - s)
    
class Relu(ActivationFunction):
    
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    
    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)
    
class LeakyRelu(ActivationFunction):
    
    @staticmethod
    def forward(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
class Softmax(ActivationFunction):

    @staticmethod
    def forward(x):
        x_stable = x - np.max(x, axis=1, keepdims=True)
        x_stable = np.clip(x_stable, -500, 500)
        e_x = np.exp(x_stable)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def derivative(x):
        pass