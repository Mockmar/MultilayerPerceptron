import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "output_train_data"

if __name__ == "__main__":

    dict_df = {}
    for dir in os.listdir(OUTPUT_DIR):
        dict_df[dir] = pd.read_csv(os.path.join(OUTPUT_DIR, dir, "train_data.csv"))

    plt.figure(figsize=(20, 12))
    colors = sns.color_palette("tab10", len(dict_df))
    for key, df in dict_df.items():
        color=colors.pop(0)
        plt.plot(df['train_loss'], label=f"{key} - Train Loss", color=color)
        plt.plot(df['val_loss'], label=f"{key} - Validation Loss", linestyle='--', color=color)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()