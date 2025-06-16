import numpy as np

class LossFunction:
    """Base class for loss functions."""
    
    @staticmethod
    def forward(y_true, y_pred):
        """Compute the forward pass of the loss function."""
        raise NotImplementedError("Forward method not implemented.")
    
    @staticmethod
    def derivative(y_true, y_pred):
        """Compute the derivative of the loss function with respect to predictions."""
        raise NotImplementedError("Derivative method not implemented.")
    
class Mse(LossFunction):
    """Mean Squared Error (MSE) loss function."""
    
    @staticmethod
    def forward(y_true, y_pred):
        """Compute the MSE loss."""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def derivative(y_true, y_pred):
        """Compute the derivative of the MSE loss with respect to predictions."""
        return -2 * (y_true - y_pred) / y_true.size
    
class BinaryCrossEntropy(LossFunction):
    """Binary Cross-Entropy loss function."""
    
    @staticmethod
    def forward(y_true, y_pred):
        """Compute the Binary Cross-Entropy loss."""
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        """Compute the derivative of the Binary Cross-Entropy loss with respect to predictions."""
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / y_true.shape[0]
    
class ClassificationCrossEntropy(LossFunction):
    """Cross-entropy loss for multi-class classification with softmax output."""

    @staticmethod
    def forward(y_true, y_pred):
        """Compute the cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # sécurité num
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))  # sum sur classes, mean sur batch

    @staticmethod
    def derivative(y_true, y_pred):
        """Derivative of softmax + crossentropy combo (simplified)."""
        return (y_pred - y_true) / y_true.shape[0]  # car softmax derivative fusionnée
