import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy of predictions.

    Parameters:
    - y_true: np.ndarray, true labels (shape: (n_samples,) or (n_samples, n_classes))
    - y_pred: np.ndarray, predicted labels (shape: (n_samples,) or (n_samples, n_classes))

    Returns:
    - float: accuracy as a percentage
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of true labels and predicted labels must match.")

    correct_predictions = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    total_predictions = y_true.shape[0]

    return correct_predictions / total_predictions