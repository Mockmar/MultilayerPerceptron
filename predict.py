import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Neuron.NeuronNetwork import Model
from fonctions.activation_function import Sigmoid, LeakyRelu, Softmax
from fonctions.loss_function import BinaryCrossEntropy, ClassificationCrossEntropy
from preprocess.OneHotEncoder import OneHotEncoder
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a trained neural network model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--data", type=str, required=True, help="Path to the data CSV file for prediction.")
    args = parser.parse_args()

    model = Model.load(args.model)

    df = pd.read_csv(args.data, header=None)
    X = df.drop(columns=30).to_numpy()
    Y = df[[30]].to_numpy()
    if isinstance(model.layers[-1].activation_function, Softmax):
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(Y)

    print(f"Loaded model: {model}")

    Y_pred = model.forward(X)
    loss_fonction = BinaryCrossEntropy()
    loss = loss_fonction.forward(Y, Y_pred)
    print(f"Loss: {loss}")

    Y_pred = model.predict(X)
    if isinstance(model.layers[-1].activation_function, Softmax):
        Y = encoder.inverse_transform(Y)

    print("Predictions:")
    print(Y_pred.astype(float))
    print("True labels:")
    print(Y)
