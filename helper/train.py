import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.manifold import Isomap
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from helper.noise import add_attribute_noise, add_class_noise
from helper.plot import plot_wrong_predictions
from copy import deepcopy

def get_onehot(class_count, class_ind):
    rez = np.zeros(class_count)
    rez[class_ind] = 1
    return rez

def train_model_k_fold(model, k, X, y, is_one_hot = False):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    unique_vals = len(np.unique(y))
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if is_one_hot:
            y_train = y_train.apply(lambda x: get_onehot(unique_vals, x))
            y_train = np.vstack(y_train).astype(np.float32)
        
        model.fit(X_train, y_train)  # Train
        y_pred = model.predict(X_test)  # Predict
        acc = accuracy_score(y_test, y_pred)  # Evaluate
        scores.append(acc)
    return model, np.average(scores)

def train_models_with_attribute_noise(X, y, models, noise_schedule=[0, 0.1, 0.2, 0.4, 0.5], dim_red = 2, transforms = None):
    print(f"Training {len(models)} models for {len(noise_schedule)} noise schedules with 5 folds => {len(models) * len(noise_schedule) * 5} trains")

    results = []
    accuracy_losses = []
    iso = Isomap(n_components=dim_red, n_jobs=4)

    fig, axes = plt.subplots(len(models), len(noise_schedule), figsize=(10*len(models), 10 * len(noise_schedule)), dpi = 200)
    trained_models = []
    for i, model_data in enumerate(models):

        result = []
        accuracy_loss = [0]
        model, model_name = model_data
        single_trained_model = []
        for j, noise_level in enumerate(noise_schedule):
            
            print(f"\tAdding attribute noise, {noise_level*100}% ...")
            X_n = add_attribute_noise(X, noise_level, [])
            y_n = y.copy()
            
            for transform in transforms:
                print(f"\tApplying transform: {transform} ...")
                X_n, y_n = transform(X_n, y_n)

            print(f"\tReducing to 2 dimensions ...")
            X_lower = iso.fit_transform(X_n)
            if X_lower.shape[0] == X.shape[0]:
                X_lower = pd.DataFrame(X_lower, index=X.index, columns=["component_1", "component_2"])
            else:
                X_lower = pd.DataFrame(X_lower, index=X_n.index, columns=["component_1", "component_2"])

            print(f"\tTraining model {model} with 5 folds...")
            trained_model, acc = train_model_k_fold(model, 5, X_n, y_n, False)
            single_trained_model.append(deepcopy(trained_model))
            print(f"\t{model} finished training...")
            print("-"*20)
            
            y_pred = trained_model.predict(X_n)
            result.append(acc)
            if j > 0:
                accuracy_loss.append((acc - result[0])/result[0] * 100)

            if len(models) > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]

            plot_wrong_predictions(trained_model, X_n, y_n, ax, X_lower)
            ax.set_title(f"{model_name} - Noise {noise_level}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        trained_models.append(single_trained_model)
        results.append((model_name, result))
        accuracy_losses.append((model_name, accuracy_loss))

    plt.show()
    return results, accuracy_losses, trained_models

def plot_model_performance_heatmap(performance_data, title, noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], cmap='inferno'):
    model_performance = {}
    for model, perf in performance_data:
        model_performance[model]= perf
    
    # Convert to 2D array
    model_labels = list(model_performance.keys())
    performance_array = np.array(list(model_performance.values()))

    ax = sns.heatmap(performance_array, annot=True, fmt=".4f", cmap=cmap, xticklabels=noise_levels, yticklabels=model_labels)
    ax.set_aspect("equal")
    
    plt.xlabel("Noise Level")
    plt.ylabel("Model")
    plt.title(title)
    plt.show()

def train_models_with_class_noise(X, y, models, noise_schedule=[0, 0.1, 0.2, 0.4, 0.5], dim_red = 2, transforms=None):
    print(f"Training {len(models)} models for {len(noise_schedule)} noise schedules with 5 folds => {len(models) * len(noise_schedule) * 5} trains")

    results = []
    accuracy_losses = []
    iso = Isomap(n_components=dim_red)

    fig, axes = plt.subplots(len(models), len(noise_schedule), figsize=(10*len(models), 10 * len(noise_schedule)), dpi = 200)
    trained_models = []
    for i, model_data in enumerate(models):

        result = []
        accuracy_loss = [0]
        model, model_name = model_data
        single_trained_model = []
        for j, noise_level in enumerate(noise_schedule):
            
            print(f"\tAdding label noise, {noise_level*100}% ...")
            X_n = X.copy()
            y_n = add_class_noise(y, noise_level)
            
            for transform in transforms:
                print(f"\tApplying transform: {transform} ...")
                X_n, y_n = transform(X_n, y_n)

            print(f"\tReducing to 2 dimensions ...")
            X_lower = iso.fit_transform(X_n)
            if X_lower.shape[0] == X.shape[0]:
                X_lower = pd.DataFrame(X_lower, index=X.index, columns=["component_1", "component_2"])
            else:
                X_lower = pd.DataFrame(X_lower, index=X_n.index, columns=["component_1", "component_2"])

            print(f"\tTraining model {model} with 5 folds...")
            trained_model, acc = train_model_k_fold(model, 5, X_n, y_n, False)
            single_trained_model.append(deepcopy(trained_model))
            print(f"\t{model} finished training...")
            print("-"*20)
            
            y_pred = trained_model.predict(X_n)
            result.append(acc)
            if j > 0:
                accuracy_loss.append((acc - result[0])/result[0] * 100)

            if len(models) > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]

            plot_wrong_predictions(trained_model, X_n, y_n, ax, X_lower)
            ax.set_title(f"{model.__class__.__name__} - Noise {noise_level}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        trained_models.append(single_trained_model)
        results.append((model_name, result))
        accuracy_losses.append((model_name,accuracy_loss))

    plt.show()
    return results, accuracy_losses, trained_models

def get_model_title(transforms):
    title=""
    for tranform in transforms:
        title += tranform.__class__.__name__ + ", "
    return title[:-2]

def run_models(X, y, models, accuracies, accuracy_changes, transforms=None, title=""):
    noise_schedule = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    acc, acc_change, trained_att_noise = train_models_with_attribute_noise(X, y, models, noise_schedule, transforms=transforms)
    accuracies.append((f"Accuracy with attribute noise {title}",acc))
    accuracy_changes.append((f"Accuracy change with attribute noise {title}",acc))

    plot_model_performance_heatmap(acc, f"Accuracy with attribute noise, {title}", noise_levels=noise_schedule)
    plot_model_performance_heatmap(acc_change, f"Accuracy change with attribute noise, {title}", noise_levels=noise_schedule)

    acc, acc_change, trained_class_noise = train_models_with_class_noise(X, y, models, noise_schedule, transforms=transforms)
    accuracies.append((f"Accuracy with label noise {title}",acc))
    accuracy_changes.append((f"Accuracy change with label noise {title}",acc))

    plot_model_performance_heatmap(acc, f"Accuracy with label noise, {title}", noise_levels=noise_schedule)
    plot_model_performance_heatmap(acc_change, f"Accuracy change with label noise, {title}", noise_levels=noise_schedule)

    return acc, acc_change, trained_att_noise, trained_class_noise

def test_models(X, y, models, noise_type="attribute"):
    #for each model
    #   run it on X and measure accuracy
    noise_schedule = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for i, model in enumerate(models):
        y_pred = model.predict(X)  # Predict
        acc = accuracy_score(y, y_pred)  # Evaluate
        print(f"Accuracy for model {model} trained with {noise_schedule[i]*100}% of {noise_type} noise: {acc}")

def difficulty_split(X_train, y_train, X_valid, y_valid, models, difficulty_categories=3):
    wrong_prediction_count = np.zeros(len(y_valid))

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        wrong_prediction_count += (y_pred != y_valid).astype(int)


    bins = np.linspace(0, len(models), num=difficulty_categories-1)
    difficulty_labels = np.digitize(wrong_prediction_count, bins, right=False)

    # Map difficulty labels to indices
    difficulty_indices = {i: np.where(difficulty_labels == i)[0] for i in np.unique(difficulty_labels)}

    return wrong_prediction_count, difficulty_indices


