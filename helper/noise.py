import pandas as pd
import numpy as np

def generate_random(x, noise_chance):
    "Returns x or a uniform random value. Noise_chance represents the chance of getting a random value"
    throw = np.random.rand()
    if throw < noise_chance:
        return np.random.rand()
    return x

def generate_gauss_random(x, noise_chance):
    "Returns x or a gaussian random value. Noise_chance represents the chance of getting a random value"
    throw = np.random.rand()
    if throw < noise_chance:
        return np.random.randn()
    return x

def add_attribute_noise(X: pd.DataFrame, noise_perc:float, class_labels, noise_gen = None):
    """noise_perc percentage of values will be replaced by a random value"""
    X_new = X.copy()
    for i, col in enumerate(X.columns):
        if col in class_labels:
            continue
        X_new[col] = X_new[col].apply(lambda x: generate_random(x, noise_perc) if noise_gen is None else noise_gen(x, noise_perc))
    return X_new

def get_random_class(y, noise_chance, classes):
    "Returns y or a random class. Noise_chance represents the chance of getting a random value"
    throw = np.random.rand()
    if throw < noise_chance:
        i = np.random.randint(0, len(classes))
        return classes[i] if classes[i]!=y else classes[(i+1)%len(classes)]
    return y

def add_class_noise(y:pd.DataFrame, noise_perc:float):
    "noise_perc percentage of labels will be randomly swapped with another label"
    y_new = y.copy()
    unique_classes = list(np.unique(y_new))
    y_new = y.apply(lambda x: get_random_class(x, noise_perc, unique_classes))
    return y_new