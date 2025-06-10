import numpy as np
from collections import defaultdict

def _validate_inputs(X, y, test_size):
    X = np.array(X)
    if y is not None:
        y = np.array(y)
        if len(X) != len(y):
            raise ValueError("X et y doivent avoir le même nombre d'échantillons.")
    n_samples = len(X)

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Si test_size est un float, il doit être entre 0 et 1.")
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise ValueError("test_size doit être un float ou un int.")

    if not 0 < n_test < n_samples:
        raise ValueError("test_size doit être > 0 et < nombre d'échantillons")

    return X, y, n_samples, n_test

def _shuffle_indices(n_samples, random_state):
    rng = np.random.default_rng(seed=random_state)
    return rng.permutation(n_samples)

def _stratified_split(y, n_test, random_state):
    label_to_indices = defaultdict(list)

    for idx, label in enumerate(y):
        if isinstance(label, np.ndarray):
            if label.ndim == 0:
                label = label.item()
            elif label.shape == (1,):
                label = label[0]
            else:
                raise ValueError("Les labels doivent être des scalaires ou des tableaux de forme (1,)")
        label_to_indices[label].append(idx)

    rng = np.random.default_rng(seed=random_state)
    train_idx, test_idx = [], []

    for indices in label_to_indices.values():
        indices = np.array(indices)
        rng.shuffle(indices)
        n_label_test = int(len(indices) * (n_test / len(y)))
        test_idx.extend(indices[:n_label_test])
        train_idx.extend(indices[n_label_test:])

    return np.array(train_idx), np.array(test_idx)

def _split_data(X, y, train_idx, test_idx):
    X_train, X_test = X[train_idx], X[test_idx]
    if y is not None:
        y_train, y_test = y[train_idx], y[test_idx]
        return X_train, X_test, y_train, y_test
    return X_train, X_test

def train_test_split(X, y=None, test_size=0.25, shuffle=True, random_state=None, stratify=None):
    X, y, n_samples, n_test = _validate_inputs(X, y, test_size)

    if stratify is not None:
        stratify = np.array(stratify)
        if len(stratify) != n_samples:
            raise ValueError("stratify doit avoir la même taille que X")
        train_idx, test_idx = _stratified_split(stratify, n_test, random_state)
    else:
        indices = np.arange(n_samples)
        if shuffle:
            indices = _shuffle_indices(n_samples, random_state)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    return _split_data(X, y, train_idx, test_idx)
