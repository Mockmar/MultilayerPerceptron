import numpy as np

class ActivationFunction:
    """Base class for activation functions."""
    
    @staticmethod
    def forward(x):
        """Compute the forward pass of the activation function."""
        raise NotImplementedError("Forward method not implemented.")
    
    @staticmethod
    def derivative(x):
        """Compute the derivative of the activation function."""
        raise NotImplementedError("Derivative method not implemented.")
    

class Sigmoid(ActivationFunction):
    """Sigmoid activation function."""
    
    @staticmethod
    def forward(x):
        """Compute the sigmoid of x."""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        """Compute the derivative of the sigmoid function."""
        s = Sigmoid.forward(x)
        return s * (1 - s)
    
class Relu(ActivationFunction):
    """ReLU activation function."""
    
    @staticmethod
    def forward(x):
        """Compute the ReLU of x."""
        return np.maximum(0, x)
    
    @staticmethod
    def derivative(x):
        """Compute the derivative of the ReLU function."""
        return np.where(x > 0, 1, 0)
    
class LeakyRelu(ActivationFunction):
    """Leaky ReLU activation function."""
    
    @staticmethod
    def forward(x, alpha=0.01):
        """Compute the Leaky ReLU of x."""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def derivative(x, alpha=0.01):
        """Compute the derivative of the Leaky ReLU function."""
        return np.where(x > 0, 1, alpha)
    
class Softmax(ActivationFunction):
    """Softmax activation function."""
    
    @staticmethod
    def forward(x):
        """Compute the softmax of vector x."""
        e_x = np.exp(x - np.max(x))  # for numerical stability
        return e_x / e_x.sum(axis=0, keepdims=True)
    
    @staticmethod
    def derivative(x):
        """Compute the derivative of the softmax function."""
        s = Softmax.forward(x)
        return np.diagflat(s) - np.outer(s, s)