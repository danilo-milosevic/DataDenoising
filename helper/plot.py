import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import Isomap
import math


def plot_attributes(
    df: pd.DataFrame,
    label_column: str,
    is_onehot: bool = False,
    figsize: tuple[int, int] = (8, 10),
    n_attrs: int = 7,
):
    """
    df - dataframe to plot
    label_column - name of the column with class labels
    is_onehot - true if class labels are onehot
    figsize - size of the plot, 8x10 by default
    n_attrs - number of attributes
    """
    column_count = len(df.columns)
    if is_onehot:
        column_count -= 1
    n_rows = int(math.sqrt(column_count))
    n_columns = column_count // n_rows
    if n_rows * n_columns < column_count:
        n_columns += 1

    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize)

    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        if col == label_column and is_onehot:
            continue
        sns.histplot(
            df[col], bins=n_attrs if col == label_column else 10, kde=True, ax=axes[i]
        )
        axes[i].set_title(f"{col}")

    plt.tight_layout(rect=[0, 0, 1.1, 1.1])
    plt.show()


def plot_attr_label(
    X: pd.DataFrame, y: pd.DataFrame, figsize: tuple[int, int] = (8, 10)
):
    """
    Plots relationship between each attribute and the class
    X - instance attribute values
    y - class labels
    figsize - size of the plot, 8x10 by default
    """
    column_count = len(X.columns)
    n_rows = int(math.sqrt(column_count))
    n_columns = column_count // n_rows
    if n_rows * n_columns < column_count:
        n_columns += 1

    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize)
    fig.suptitle("Scatter Plots of Attributes vs Class_Label", fontsize=12)

    axes = axes.flatten()

    for i, col in enumerate(X.columns):
        sns.scatterplot(x=X[col], y=y, alpha=0.6, ax=axes[i])
        axes[i].set_title(f"{col}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_wrong_predictions(model, X, y_true, ax, X_lowered):
    y_pred = model.predict(X)

    # Identify misclassified and correctly classified indices
    misclassified_idx = (y_pred != y_true).to_numpy().nonzero()[0]
    correctly_classified_idx = (y_pred == y_true).to_numpy().nonzero()[0]

    # Scatter plot correctly classified points (semi-transparent circles)
    ax.scatter(
        X_lowered.iloc[correctly_classified_idx, 0],
        X_lowered.iloc[correctly_classified_idx, 1],
        color="blue",
        alpha=0.5,
        edgecolors="k",
        label="Correctly Classified",
    )

    # Scatter plot misclassified points (red X markers)
    ax.scatter(
        X_lowered.iloc[misclassified_idx, 0],
        X_lowered.iloc[misclassified_idx, 1],
        color="red",
        marker="x",
        label="Misclassified",
    )

    ax.legend()
    ax.set_title("Classification Results in Reduced Space")


def plot_difficulties(X, indices_map):
    iso = Isomap(n_components=2, n_jobs=-1)
    X_lowered = iso.fit_transform(X)
    color_map = {0: "green", 1: "orange", 2: "red"}
    for difficulty in list(indices_map.keys()):
        idxs = indices_map[difficulty]
        color = color_map[difficulty]
        sns.scatterplot(
            X_lowered[idxs],
            color=color,
            alpha=0.5,
            edgecolors="k",
            label=f"Difficulty {difficulty}",
        )
