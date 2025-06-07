import pandas as pd
import matplotlib
matplotlib.use('gtk3agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

OUTPUT_DIR = "output_dir_analyse"

def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset(path):
    df = pd.read_csv(path, header=None)
    print("Initial DataFrame:")
    df.info()
    return df

def save_describe(df):
    describe = df.describe()
    describe_path = os.path.join(OUTPUT_DIR, "describe.csv")
    describe.to_csv(describe_path)
    print("Data description saved to:", describe_path)

def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop(columns=0, inplace=True)
    x = df.drop(columns=1)
    y = df[[1]]
    return x, y

def plot_correlation_matrix(x):
    corr_matrix = x.corr().abs()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    plt.savefig(corr_path)
    print("Correlation matrix saved to:", corr_path)
    plt.show()

def plot_feature_distributions(x, y):
    # Fusionner pour hue
    df_plot = pd.concat([x, y], axis=1)
    df_plot.columns = list(range(df_plot.shape[1]))
    feature_columns = df_plot.columns[:-1]
    class_column = df_plot.columns[-1]

    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()

    for i, feature in enumerate(feature_columns):
        sns.kdeplot(data=df_plot, x=feature, hue=class_column, ax=axes[i], common_norm=False)
        axes[i].set_title(f"Feature {feature}")

    # Supprimer les axes vides
    for i in range(len(feature_columns), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Distribution des 30 features selon la classe", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dist_path = os.path.join(OUTPUT_DIR, "feature_distributions.png")
    plt.savefig(dist_path)
    print("Feature distributions saved to:", dist_path)
    plt.show()

def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Usage: python analyse.py <path_dataset>")
        
        dataset_path = sys.argv[1]

        ensure_output_directory()
        df = load_dataset(dataset_path)
        save_describe(df)

        X, Y = preprocess_data(df)
        plot_correlation_matrix(X)
        plot_feature_distributions(X, Y)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
