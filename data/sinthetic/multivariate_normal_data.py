import numpy as np


def generate_multivariate_normal_data(n_samples, n_features, centers, cluster_std, n_datasets=10):
    datasets = []
    for _ in range(n_datasets):
        X = []
        y = []
        for i, center in enumerate(np.random.rand(centers, n_features) * 10):
            points = np.random.multivariate_normal(center, np.eye(
                n_features) * cluster_std, n_samples // centers)
            X.extend(points)
            y.extend([i] * (n_samples // centers))
        datasets.append((np.array(X), np.array(y)))
    return datasets
