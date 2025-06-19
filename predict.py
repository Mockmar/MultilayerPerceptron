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
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data CSV file for prediction.")
    args = parser.parse_args()

    # Load the model
    model = Model()
    model.load(args.model_path)

    # Load the data
    df = pd.read_csv(args.data_path, header=None)
    X = df.drop(columns=30).to_numpy()
    Y = df[[30]].to_numpy()
    
    if model.layers[-1].output_size == 2:
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(Y)