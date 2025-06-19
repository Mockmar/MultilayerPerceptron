import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Neuron.NeuronNetwork import Model
from fonctions.activation_function import Sigmoid, LeakyRelu, Softmax
from fonctions.loss_function import BinaryCrossEntropy, ClassificationCrossEntropy
from preprocess.OneHotEncoder import OneHotEncoder
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    parser.add_argument("--layers", type=list, default=[24, 24, 6, 2],
                        help="List of layer sizes for the model.")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--train_data", type=str, default="data/train_set.csv",
                        help="Path to the training data CSV file.")
    parser.add_argument("--val_data", type=str, default="data/val_set.csv",
                        help="Path to the validation data CSV file.")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_data, header=None)
    val_df = pd.read_csv(args.val_data, header=None)

    X_train = train_df.drop(columns=30)
    X_train = X_train.to_numpy()
    Y_train = train_df[[30]]
    Y_train = Y_train.to_numpy()

    X_val = val_df.drop(columns=30)
    X_val = X_val.to_numpy()
    Y_val = val_df[[30]]
    Y_val = Y_val.to_numpy()

    if args.layers[-1] not in [1, 2]:
        raise ValueError("Last layer must have 1 or 2 neurons for classification.")
    if args.layers[-1] == 1:
        activation_function = Sigmoid()
        loss_function = BinaryCrossEntropy()
    else:
        activation_function = Softmax()
        loss_function = ClassificationCrossEntropy()
        encoder = OneHotEncoder()

        Y_train = encoder.fit_transform(Y_train)
        Y_val = encoder.transform(Y_val)

    model = Model()
    model.add_layer(output_size=args.layers[0], activation_function=LeakyRelu(), input_size=X_train.shape[1])
    for i in args.layers[1:-1]:
        model.add_layer(output_size=i, activation_function=LeakyRelu())
    model.add_layer(output_size=args.layers[-1], activation_function=activation_function)
    model.compile(loss_function=loss_function, learning_rate=args.learning_rate)

    model.train((X_train, Y_train), (X_val, Y_val), epochs=args.epochs, verbose=True, batch_size=args.batch_size)

