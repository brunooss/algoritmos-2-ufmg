from sklearn.datasets import make_blobs, make_moons


def generate_sklearn_datasets(n_samples=1000, n_features=2, centers=4, cluster_std=1.0, n_datasets=10):
    datasets = []

    for _ in range(5):
        datasets.append(make_blobs(
            n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std))
        datasets.append(make_moons(n_samples=n_samples, noise=0.1))

    return datasets
