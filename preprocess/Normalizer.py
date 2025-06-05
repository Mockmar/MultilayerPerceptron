import json

class Normalizer:
    def __init__(self):
        self.mean = []
        self.std = []

    def fit(self, array):
        nb_rows, nb_columns = array.shape
        if nb_rows == 0 or nb_columns == 0:
            raise ValueError("Le tableau ne doit pas être vide.")
        for i in range(nb_columns):
            if array[:, i].dtype == 'object':
                continue
            mean = array[:, i].mean()
            std = array[:, i].std()
            if std == 0:
                raise ValueError(f"La colonne {i} a une variance nulle, normalisation impossible.")
            array[:, i] = (array[:, i] - mean) / std
            self.mean.append(mean)
            self.std.append(std)

    def transform(self, array):
        if not self.mean or not self.std:
            raise ValueError("Le normaliseur n'a pas été ajusté. Appelez fit() d'abord.")
        nb_rows, nb_columns = array.shape
        if len(self.mean) != nb_columns or len(self.std) != nb_columns:
            raise ValueError("Le tableau à normaliser a un nombre de colonnes différent de celui utilisé pour l'ajustement.")
        for i in range(nb_columns):
            if array[:, i].dtype == 'object':
                continue
            array[:, i] = (array[:, i] - self.mean[i]) / self.std[i]
        return array
    
    def fit_transform(self, array):
        self.fit(array)
        return self.transform(array)
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({'mean': self.mean, 'std': self.std}, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        if not data['std'] or not data['mean']:
            raise ValueError("Le fichier de normalisation est vide ou mal formé.")
        if len(data['mean']) != len(data['std']):
            raise ValueError("Le nombre de moyennes et d'écarts-types ne correspond pas.")
        self.std = data['std']
        self.mean = data['mean']