import numpy as np
import sklearn.cluster as skcluster
import time


def calculate_max_radius(X, centroids, labels):
    max_radius = 0
    for i, centroid in enumerate(centroids):
        distances = np.linalg.norm(X[labels == i] - centroid, axis=1)
        cluster_radius = distances.max()
        if cluster_radius > max_radius:
            max_radius = cluster_radius
    return max_radius


def run_k_means(X, k, **kwargs):
    """
    Executa o algoritmo K-Means do scikit-learn e retorna os resultados e a duração do algoritmo.

    Parâmetros:
    X (array): Dados de entrada (n amostras x m características).
    k (int): Número de clusters.
    **kwargs: Argumentos adicionais para passar à função `k_means` do scikit-learn.

    Retorna:
    tuple: (centroids, labels, number, duration)
        - centroids: Centróides dos clusters.
        - labels: Rótulos dos clusters para cada amostra.
        - number: Número de iterações realizadas.
        - duration: Tempo de execução do algoritmo em segundos.
    """
    start_time = time.time()

    centroids, labels, _ = skcluster.k_means(X, k, **kwargs)

    max_radius = calculate_max_radius(X, centroids, labels)

    duration = time.time() - start_time

    return max_radius, labels, duration
