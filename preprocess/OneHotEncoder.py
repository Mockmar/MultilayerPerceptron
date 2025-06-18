import numpy as np

class OneHotEncoder:
    def __init__(self):
        pass

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.index_to_class = {idx: cls for idx, cls in enumerate(self.classes_)}

    def transform(self, y: np.ndarray):
        if not hasattr(self, 'classes_'):
            raise ValueError("The encoder has not been fitted yet. Call 'fit' with the labels first.")
        
        one_hot = np.zeros((len(y), len(self.classes_)), dtype=int)
        for i, label in enumerate(y):
            if label in self.class_to_index:
                one_hot[i, self.class_to_index[label]] = 1
            else:
                raise ValueError(f"Label '{label}' not found in fitted classes.")
        return one_hot
    
    def fit_transform(self, y: np.ndarray):
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, one_hot):
        if one_hot.shape[1] != len(self.classes_):
            raise ValueError("Input shape does not match the number of classes.")
        indices = np.argmax(one_hot, axis=1)
        return np.array([self.index_to_class[idx] for idx in indices])