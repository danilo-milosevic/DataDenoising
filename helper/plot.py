import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

def plot_attributes(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 5, figsize=(20, 20))
    fig.suptitle("Histograms of Attributes", fontsize=10)

    axes = axes.flatten()

    for i, col in enumerate(df.columns):  
        if 'OneHot' in col:
            continue
        sns.histplot(df[col], bins= 7 if col == 'Class_Label' else 10, kde=True, ax=axes[i])
        axes[i].set_title(f'{col}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_attr_label(df: pd.DataFrame, labels = None):
    fig, axes = plt.subplots(4, 4, figsize=(15, 12))
    fig.suptitle("Scatter Plots of Attributes vs Class_Label", fontsize=16)

    axes = axes.flatten()

    for i, col in enumerate(df.columns[:-2]):  
        sns.scatterplot(x=df[col], y= df['Class_Label'] if labels is None else labels, alpha=0.6, ax=axes[i])
        axes[i].set_title(f'{col}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_wrong_predictions(model, X, y_true, ax, X_lowered):
    y_pred = model.predict(X)

    # Identify misclassified and correctly classified indices
    misclassified_idx = (y_pred != y_true).to_numpy().nonzero()[0]
    correctly_classified_idx = (y_pred == y_true).to_numpy().nonzero()[0]

    # Scatter plot correctly classified points (semi-transparent circles)
    ax.scatter(X_lowered.iloc[correctly_classified_idx, 0], 
               X_lowered.iloc[correctly_classified_idx, 1], 
               color='blue', alpha=0.5, edgecolors='k', label='Correctly Classified')

    # Scatter plot misclassified points (red X markers)
    ax.scatter(X_lowered.iloc[misclassified_idx, 0], 
               X_lowered.iloc[misclassified_idx, 1], 
               color='red', marker='x', label='Misclassified')

    ax.legend()
    ax.set_title("Classification Results in Reduced Space")