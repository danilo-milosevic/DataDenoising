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

def transform_pca(X, y, n_comp = 0.98):
    pca = PCA(n_components=n_comp)
    X_t_pca = pca.fit_transform(X)
    print(f"Reduced from {len(X.iloc[0])} to {pca.n_components_} dimensions")
    X_t_pca = pd.DataFrame(X_t_pca, columns=[f'PC{i+1}' for i in range(X_t_pca.shape[1])])
    X_t_pca.index = X.index
    return X_t_pca, y

def remove_outliers_zscore(X, y, threshold=3):
    """
    Z-score based outlier removal (index-safe).
    Returns:
        X_no_outliers, y_no_outliers, indices_outliers (index labels)
    """
    # Compute z-scores row-wise
    zs = zscore(X, axis=1)

    # Preserve index and columns
    zs = pd.DataFrame(zs, index=X.index, columns=X.columns)

    # Max absolute z-score per row
    max_z = zs.abs().max(axis=1)

    # Outlier indices (LABELS)
    indices_outliers = max_z[max_z >= threshold].index.tolist()

    # Drop using label-based indexing
    X_no_outliers = X.drop(index=indices_outliers)
    y_no_outliers = y.drop(index=indices_outliers)

    return X_no_outliers, y_no_outliers, indices_outliers


def remove_outliers_isf(X, y):
    """
    Isolation Forest based outlier removal (index-safe).
    Returns:
        X_no_outliers, y_no_outliers, indices_outliers (index labels)
    """
    isf = IsolationForest(
        n_estimators=100,
        random_state=42,
        max_features=1.0,
        contamination=0.2
    )

    preds = isf.fit_predict(X)

    # Convert positional indices → index labels
    indices_outliers = X.index[preds == -1].tolist()

    # Drop using label-based indexing
    X_no_outliers = X.drop(index=indices_outliers)
    y_no_outliers = y.drop(index=indices_outliers)

    return X_no_outliers, y_no_outliers, indices_outliers


def remove_outliers_db(X, y):
    """
    DBSCAN based outlier removal (index-safe).
    Returns:
        X_no_outliers, y_no_outliers, indices_outliers (index labels)
    """
    db = DBSCAN(eps=0.2, min_samples=2, n_jobs=-1)
    preds = db.fit_predict(X)

    # Convert positional indices → index labels
    indices_outliers = X.index[preds == -1].tolist()

    # Drop using label-based indexing
    X_no_outliers = X.drop(index=indices_outliers)
    y_no_outliers = y.drop(index=indices_outliers)

    return X_no_outliers, y_no_outliers, indices_outliers

def bin_attributes_mean(X, y):
    bin_counts = 10
    X_binned = X.copy()
    bin_edges = np.linspace(0, 1, bin_counts + 1)
    bin_labels = np.linspace(0, 1, bin_counts)
    
    for col in X_binned.columns:
        bins = pd.cut(X_binned[col], bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_map = y.groupby(bins).mean().to_dict()
        # Convert to float first, then fillna
        X_binned[col] = bins.map(bin_map).astype(float).fillna(y.mean())
    
    return X_binned, y

def bin_attributes_median(X, y):
    bin_counts = 10
    X_binned = X.copy()
    bin_edges = np.linspace(0, 1, bin_counts + 1)
    bin_labels = np.linspace(0, 1, bin_counts)
    
    for col in X_binned.columns:
        bins = pd.cut(X_binned[col], bins=bin_edges, labels=bin_labels, include_lowest=True)
        bin_map = y.groupby(bins).median().to_dict()
        # Convert to float first, then fillna
        X_binned[col] = bins.map(bin_map).astype(float).fillna(y.median())
    
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


def remove_label_noise_ensemble_filter(X, y, 
                                       n_splits = 5, 
                                       voting_threshold = 0.5, 
                                       classifiers = [
                                        RandomForestClassifier(),
                                        SVC(probability=True),
                                        GradientBoostingClassifier(),
                                        KNeighborsClassifier(),
                                    ]):
    """
    Ensemble-based label noise removal (index-safe).
    Returns:
        X_no_noise, y_no_noise, indices_noise (index labels)
    """

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
    
    # Convert positional indices → index labels
    indices_noise = X.index[noise_instances].tolist()
    
    # Drop using label-based indexing
    X_no_noise = X.drop(index=indices_noise)
    y_no_noise = y.drop(index=indices_noise)

    return X_no_noise, y_no_noise, indices_noise


def remove_label_noise_cross_validated_committees_filter(X, y):
    """
    Cross-validated committees based label noise removal (index-safe).
    Returns:
        X_no_noise, y_no_noise, indices_noise (index labels)
    """
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
    
    # Convert positional indices → index labels
    indices_noise = X.index[noise_instances].tolist()
    
    # Drop using label-based indexing
    X_no_noise = X.drop(index=indices_noise)
    y_no_noise = y.drop(index=indices_noise)

    return X_no_noise, y_no_noise, indices_noise


def remove_label_noise_iterative_partitioning_filter(X, y):
    """
    Iterative partitioning based label noise removal (index-safe).
    Returns:
        X_no_noise, y_no_noise, indices_noise (index labels)
    """
    X_filtered = X.copy()
    y_filtered = y.copy()
    removed_indices = []

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
        
        # Track removed index labels before filtering
        indices_to_remove = X_filtered.index[~keep_instances].tolist()
        removed_indices.extend(indices_to_remove)
        
        # Drop using label-based indexing
        X_filtered = X_filtered.drop(index=indices_to_remove)
        y_filtered = y_filtered.drop(index=indices_to_remove)

    return X_filtered, y_filtered, removed_indices
