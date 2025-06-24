import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of true labels and predicted labels must match.")

    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = y_true.shape[0]

    return correct_predictions / total_predictions

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of true labels and predicted labels must match.")

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    if true_positive + false_negative == 0:
        return 0.0

    return true_positive / (true_positive + false_negative)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of true labels and predicted labels must match.")

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))

    if true_positive + false_positive == 0:
        return 0.0

    return true_positive / (true_positive + false_positive)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of true labels and predicted labels must match.")

    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)