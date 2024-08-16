from data.sinthetic.multivariate_normal_data import generate_multivariate_normal_data
from data.sinthetic.scikit_examples_data import generate_sklearn_datasets


def generate_datasets():
    """
    Gera 30 conjuntos de dados sintéticos com distribuição normal multivariada.

    Retorna:
    list: Lista com os conjuntos de dados gerados.
    """
    n_samples = 1000
    n_features = 2
    centers = 4
    cluster_std = 1.0
    n_datasets = 10

    datasets_sklearn = generate_sklearn_datasets(
        n_samples, n_features, centers, cluster_std, n_datasets)

    datasets_multivariate = generate_multivariate_normal_data(
        n_samples, n_features, centers, cluster_std, n_datasets)

    datasets = datasets_sklearn + datasets_multivariate
    n_clusters = [5] * n_datasets + [5] * n_datasets

    return datasets, n_clusters
