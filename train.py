import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Neuron.NeuronNetwork import Model
from fonctions.activation_function import Sigmoid, LeakyRelu, Softmax
from fonctions.metrics import accuracy, recall, precision, f1_score
from fonctions.loss_function import BinaryCrossEntropy, ClassificationCrossEntropy
from preprocess.OneHotEncoder import OneHotEncoder
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    parser.add_argument("--layers", type=int, nargs='+', default=[24, 24, 6, 2],
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
    parser.add_argument("--save_model", action='store_true', default=False,
                        help="Whether to save the trained model.")
    parser.add_argument("--no_early_stop", action='store_false', default=True,
                        help="Whether to use early stopping during training.")
    parser.add_argument("--no_save_data", action='store_false', default=True,
                        help="Whether to save training data and plots.")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_data, header=None)
    val_df = pd.read_csv(args.val_data, header=None)

    X_train = train_df.drop(columns=30)
    X_train = X_train.to_numpy()
    Y_train = train_df[[30]]
    Y_train = Y_train.to_numpy()
    Y_train_save = Y_train.copy()

    X_val = val_df.drop(columns=30)
    X_val = X_val.to_numpy()
    Y_val = val_df[[30]]
    Y_val = Y_val.to_numpy()
    Y_val_save = Y_val.copy()

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

    model.train((X_train, Y_train), (X_val, Y_val), epochs=args.epochs, verbose=True, batch_size=args.batch_size, early_stopping=args.no_early_stop)
    
    y_pred_val = model.predict(X_val).reshape(-1, 1)
    y_pred_train = model.predict(X_train).reshape(-1, 1)

    accu_train = accuracy(Y_train_save, y_pred_train)
    accu_val = accuracy(Y_val_save, y_pred_val)
    recall_train = recall(Y_train_save, y_pred_train)
    recall_val = recall(Y_val_save, y_pred_val)
    precision_train = precision(Y_train_save, y_pred_train)
    precision_val = precision(Y_val_save, y_pred_val)
    f1_train = f1_score(Y_train_save, y_pred_train)
    f1_val = f1_score(Y_val_save, y_pred_val)

    print("Training Accuracy:   ", accu_train)
    print("Validation Accuracy: ", accu_val)
    print("Training Recall:     ", recall_train)
    print("Validation Recall:   ", recall_val)
    print("Training Precision:  ", precision_train)
    print("Validation Precision:", precision_val)
    print("Training F1 Score:   ", f1_train)
    print("Validation F1 Score: ", f1_val)



    model_name = 'layers(' + '_'.join(map(str, args.layers)) + ')_lr(' + str(args.learning_rate) + ')'
    if args.save_model:
        path = 'models/' + model_name + '.model'
        model.save(path)

    train_loss_lst = model.train_loss_lst
    val_loss_lst = model.val_loss_lst
    train_accuracy_lst = model.train_accuracy_lst
    val_accuracy_lst = model.val_accuracy_lst

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(train_loss_lst, label='Train Loss', color='blue')
    plt.plot(val_loss_lst, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(train_accuracy_lst, label='Train Accuracy', color='blue')
    plt.plot(val_accuracy_lst, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs('output_train_data/' + model_name, exist_ok=True)
    plt.savefig('output_train_data/' + model_name + '/loss_accuracy_plot.png')
    plt.show()


    train_data = {
        'train_loss': train_loss_lst,
        'val_loss': val_loss_lst,
        'train_accuracy': train_accuracy_lst,
        'val_accuracy': val_accuracy_lst
    }

    train_data_df = pd.DataFrame(train_data)
    if args.no_save_data:
        train_data_df.to_csv('output_train_data/' + model_name + '/train_data.csv', index=False)

