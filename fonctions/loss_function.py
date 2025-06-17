import numpy as np

class LossFunction:
    
    @staticmethod
    def forward(y_true, y_pred):
        raise NotImplementedError("Forward method not implemented.")
    
    @staticmethod
    def derivative(y_true, y_pred):
        raise NotImplementedError("Derivative method not implemented.")
    
class Mse(LossFunction):
    
    @staticmethod
    def forward(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def derivative(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size
    
class BinaryCrossEntropy(LossFunction):

    @staticmethod
    def forward(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / y_true.shape[0]
    
class ClassificationCrossEntropy(LossFunction):

    @staticmethod
    def forward(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def derivative(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]
