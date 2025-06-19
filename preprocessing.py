from preprocess.Normalizer import Normalizer
from preprocess.train_test_split import train_test_split
import pandas as pd
import numpy as np
import os
import sys

# Benin B=0, Malin M=1

def encodage_classes(df):
    df[1] = df[1].apply(lambda x: 0 if x == 'B' else 1)
    return df

def split_dataset(df):
    x = df.drop(columns=1)
    y = df[[1]]

    x = np.array(x)
    y = np.array(y)

    return x, y

def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Usage: python preprocessing.py <path_to_dataset>")
        
        dataset_path = sys.argv[1]
        df = pd.read_csv(dataset_path, header=None)

        df.dropna(inplace=True)
        df.drop(columns=0, inplace=True)

        df = encodage_classes(df)
        X, Y = split_dataset(df)

        normalizer = Normalizer()
        X_normalized = normalizer.fit_transform(X)
        normalizer.save('preprocess/normalizer.json')

        X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size=0.20, stratify=Y)

        train_set = np.concatenate((X_train, Y_train), axis=1)
        test_set = np.concatenate((X_test, Y_test), axis=1)

        np.savetxt('data/train_set.csv', train_set, delimiter=',', fmt='%f')
        np.savetxt('data/val_set.csv', test_set, delimiter=',', fmt='%f')

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()