"""Clustering Module for Traffic Accident Analysis

This module implements clustering techniques (KMeans, DBSCAN) to group accidents by patterns
such as location, time, and severity. It includes functions for feature engineering, scaling,
model fitting, evaluation, and visualization-friendly summaries.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ClusteringResult:
    model_name: str
    params: dict
    n_clusters: int
    silhouette: float
    labels: np.ndarray
    centers: np.ndarray | None


def prepare_features(df, feature_columns, scale=True):
    """
    Prepare feature matrix for clustering.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (list): Columns to use as features
        scale (bool): Whether to scale features
        
    Returns:
        np.ndarray: Feature matrix X
        StandardScaler | None: Fitted scaler if scale=True else None
    """
    X = df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    return X.values, None


def run_kmeans(df, feature_columns, k=4, scale=True, random_state=42):
    X, scaler = prepare_features(df, feature_columns, scale=scale)
    
    print(f"Running KMeans with k={k} on {X.shape[0]} records and {X.shape[1]} features")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    
    result = ClusteringResult(
        model_name='KMeans',
        params={'k': k},
        n_clusters=k,
        silhouette=float(sil),
        labels=labels,
        centers=kmeans.cluster_centers_
    )
    
    print(f"KMeans silhouette score: {sil:.3f}")
    return result


def run_dbscan(df, feature_columns, eps=0.5, min_samples=5, scale=True):
    X, scaler = prepare_features(df, feature_columns, scale=scale)
    
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    
    # Filter out noise (-1) for silhouette
    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        sil = silhouette_score(X[mask], labels[mask])
    else:
        sil = -1
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    result = ClusteringResult(
        model_name='DBSCAN',
        params={'eps': eps, 'min_samples': min_samples},
        n_clusters=int(n_clusters),
        silhouette=float(sil),
        labels=labels,
        centers=None
    )
    
    print(f"DBSCAN clusters: {n_clusters}, silhouette score: {sil:.3f}")
    return result


def summarize_clusters(df, labels, group_by_cols=None):
    """
    Summarize clusters with basic statistics.
    """
    df = df.copy()
    df['cluster'] = labels
    
    print("\nCluster summary:")
    summary = df.groupby('cluster').agg(['count', 'mean', 'std'])
    print(summary)
    
    if group_by_cols:
        for col in group_by_cols:
            if col in df.columns:
                print(f"\nDistribution by {col}:")
                print(df.groupby(['cluster', col]).size().unstack(fill_value=0))
    
    return summary


def optimal_k_elbow(df, feature_columns, k_range=(2, 10), scale=True):
    """
    Compute inertia across k range to help choose k.
    """
    X, scaler = prepare_features(df, feature_columns, scale=scale)
    ks = list(range(k_range[0], k_range[1] + 1))
    inertias = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        print(f"k={k}: inertia={kmeans.inertia_:.2f}")
    
    print("\nElbow method results:")
    for k, inertia in zip(ks, inertias):
        print(f"  k={k}: inertia={inertia:.2f}")
    
    return ks, inertias


def run_clustering_pipeline(df, feature_columns, method='kmeans', k=4, eps=0.5, min_samples=5, scale=True, group_by_cols=None):
    """
    Run a full clustering pipeline and print summaries.
    """
    if method == 'kmeans':
        result = run_kmeans(df, feature_columns, k=k, scale=scale)
    elif method == 'dbscan':
        result = run_dbscan(df, feature_columns, eps=eps, min_samples=min_samples, scale=scale)
    else:
        raise ValueError("Unsupported method. Choose 'kmeans' or 'dbscan'.")
    
    summarize_clusters(df, result.labels, group_by_cols=group_by_cols)
    
    return result

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Run clustering on traffic accident data')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--features', type=str, nargs='+', required=True, help='List of feature column names')
    parser.add_argument('--method', type=str, default='kmeans', choices=['kmeans', 'dbscan'], help='Clustering method')
    parser.add_argument('--k', type=int, default=4, help='Number of clusters for KMeans')
    parser.add_argument('--eps', type=float, default=0.5, help='EPS parameter for DBSCAN')
    parser.add_argument('--min_samples', type=int, default=5, help='min_samples for DBSCAN')
    parser.add_argument('--no-scale', action='store_true', help='Disable feature scaling')
    parser.add_argument('--groupby', type=str, nargs='*', default=None, help='Columns to show distributions by cluster')
    
    args = parser.parse_args()
    scale = not args.no_scale
    
    df = pd.read_csv(args.data)
    result = run_clustering_pipeline(
        df, 
        feature_columns=args.features, 
        method=args.method, 
        k=args.k, 
        eps=args.eps, 
        min_samples=args.min_samples,
        scale=scale,
        group_by_cols=args.groupby
    )
    
    # Print JSON summary for reproducibility
    output = {
        'model': result.model_name,
        'params': result.params,
        'n_clusters': result.n_clusters,
        'silhouette': result.silhouette
    }
    print("\nJSON Summary:")
    print(json.dumps(output, indent=2))
