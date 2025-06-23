from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def transform_pca(X, y):
    pca = PCA(n_components=5)
    X_t_pca = pca.fit_transform(X)
    X_t_pca = pd.DataFrame(X_t_pca, columns=[f'PC{i+1}' for i in range(X_t_pca.shape[1])])
    X_t_pca.index = X.index
    return X_t_pca, y

def remove_outliers_zscore(X, y):
    threshold = 3
    zs = zscore(X, axis = 1)
    zs = zs.abs().max(axis=1).to_frame(name='Max')
    zs = zs[zs.abs() >= threshold ].stack()
    
    indices_outliers = list(zs.index.get_level_values(0))
    X_no_outliers = X.drop(indices_outliers)
    y_no_outliers = y.drop(indices_outliers)
    return X_no_outliers, y_no_outliers

def remove_outliers_isf(X, y):
    isf = IsolationForest(n_estimators=100, random_state=42, max_features=1.0, contamination=0.2)
    preds = isf.fit_predict(X)
    indices = np.where(preds == -1)[0]
    X_no_outliers = X.drop(X.index[indices])
    y_no_outliers = y.drop(y.index[indices])
    return X_no_outliers, y_no_outliers

def remove_outliers_db(X, y):
    db = DBSCAN(eps=0.6, min_samples=2, n_jobs=-1)
    preds = db.fit_predict(X)
    indices = np.where(preds == -1)[0]

    X_no_outliers = X.drop(X.index[indices])
    y_no_outliers = y.drop(y.index[indices])
    return X_no_outliers, y_no_outliers

def bin_attributes_mean(X, y):
    bin_counts = 10
    X_binned = X.copy()
    bin_edges = np.linspace(0, 1, bin_counts + 1)
    bin_labels = np.linspace(0, 1, bin_counts)
    
    for col in X_binned.columns:
        bins = pd.cut(X_binned[col], bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_map = X_binned.groupby(bins)[col].mean().to_dict()
        X_binned[col] = bins.map(bin_map)
    
    return X_binned, y

def bin_attributes_median(X, y):
    bin_counts = 10
    X_binned = X.copy()
    bin_edges = np.linspace(0, 1, bin_counts + 1)
    bin_labels = np.linspace(0, 1, bin_counts)
    
    for col in X_binned.columns:
        bins = pd.cut(X_binned[col], bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_map = X_binned.groupby(bins)[col].median().to_dict()
        X_binned[col] = bins.map(bin_map)
    
    return X_binned, y

def regression_reduce_noise(X, y):
    X_cleaned = X.copy()
    
    for col in X.columns:
        # Define features (all columns except the target one)
        X_other = X.drop(columns=[col])
        y_target = X[col]
        
        # Drop rows with missing values
        valid_rows = ~X_other.isnull().any(axis=1) & ~y_target.isnull()
        X_other = X_other[valid_rows]
        y_target = y_target[valid_rows]
        
        if X_other.shape[1] > 0 and len(y_target) > 1:  # Ensure we have enough data
            model = LinearRegression()
            model.fit(X_other, y_target)
            
            # Predict and replace NaN or outliers
            predicted_values = model.predict(X_other)
            X_cleaned.loc[valid_rows, col] = predicted_values
    
    return X_cleaned, y


def remove_label_noise_ensemble_filter(X, y):
    n_splits = 5
    voting_threshold = 0.5

    classifiers = [
        RandomForestClassifier(),
        SVC(probability=True),
        GradientBoostingClassifier(),
        KNeighborsClassifier(),
    ]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_instances = len(y)
    mislabel_counts = np.zeros(n_instances, dtype=int)

    for classifier in classifiers:
        misclassified = np.zeros(n_instances, dtype=int)

        for train_idx, test_idx in skf.split(X, y):
            model = clone(classifier)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict(X.iloc[test_idx])
            misclassified[test_idx] = (y_pred != y.iloc[test_idx].values)

        mislabel_counts += misclassified

    noise_instances = mislabel_counts / len(classifiers) > voting_threshold
    X_filtered = X.loc[~noise_instances].reset_index(drop=True)
    y_filtered = y.loc[~noise_instances].reset_index(drop=True)

    return X_filtered, y_filtered


def remove_label_noise_cross_validated_committees_filter(X, y):
    n_splits = 5
    voting_threshold = 0.5
    base_classifier = RandomForestClassifier()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_instances = len(y)

    classifiers = []
    for train_idx, _ in skf.split(X, y):
        model = clone(base_classifier)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        classifiers.append(model)

    misclassified = np.zeros(n_instances, dtype=int)
    for model in classifiers:
        y_pred = model.predict(X)
        misclassified += (y_pred != y.values)

    noise_instances = misclassified / n_splits > voting_threshold
    X_filtered = X.loc[~noise_instances].reset_index(drop=True)
    y_filtered = y.loc[~noise_instances].reset_index(drop=True)

    return X_filtered, y_filtered


def remove_label_noise_iterative_partitioning_filter(X, y):
    X_filtered = X.copy()
    y_filtered = y.copy()

    max_iterations = 10
    n_splits = 5
    voting_threshold = 0.5
    good_data_ratio = 0.1
    base_classifier = RandomForestClassifier()

    for _ in range(max_iterations):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        n_instances = len(y_filtered)
        classifiers = []

        for train_idx, _ in skf.split(X_filtered, y_filtered):
            model = clone(base_classifier)
            model.fit(X_filtered.iloc[train_idx], y_filtered.iloc[train_idx])
            classifiers.append(model)

        misclassified = np.zeros(n_instances, dtype=int)
        for model in classifiers:
            y_pred = model.predict(X_filtered)
            misclassified += (y_pred != y_filtered.values)

        noise_instances = misclassified / n_splits > voting_threshold
        good_instances = ~noise_instances

        if np.sum(good_instances) == 0:
            break  # Prevent empty dataset

        n_good_samples = min(int(good_data_ratio * n_instances), np.sum(good_instances))
        good_indices = np.random.choice(np.where(good_instances)[0], n_good_samples, replace=False)

        keep_instances = ~(noise_instances | np.isin(np.arange(n_instances), good_indices))

        X_filtered = X_filtered.iloc[keep_instances].reset_index(drop=True)
        y_filtered = y_filtered.iloc[keep_instances].reset_index(drop=True)

    return X_filtered, y_filtered
