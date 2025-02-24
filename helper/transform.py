from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression


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

def bin_attributes(X, y):
    bin_counts = 10
    X_binned = X.copy()
    
    for col in X_binned.columns:
        X_binned[col] = pd.cut(X_binned[col], bins=np.linspace(0, 1, bin_counts + 1), labels=np.linspace(0, 1, bin_counts), include_lowest=True)
    
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