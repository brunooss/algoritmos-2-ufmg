import numpy as np


def minkowski_dist(x, y, p=2):
    return np.sum(np.abs(x - y) ** p) ** (1/p)


def minkowski_dist(point_a, point_b, p=2):
    """
    Calcula a distância de Minkowski entre dois pontos.

    Parâmetros:
    point_a (array-like): Coordenadas do primeiro ponto.
    point_b (array-like): Coordenadas do segundo ponto.
    p (float): Parâmetro da métrica de Minkowski. p=1 corresponde à distância de Manhattan,
               p=2 corresponde à distância Euclidiana.

    Retorna:
    dist (float): Distância de Minkowski entre os dois pontos.
    """
    return np.sum(np.abs(point_a - point_b) ** p) ** (1 / p)


def pairwise_distances(points, p=2):
    """
    Calcula a matriz de distâncias entre todos os pares de pontos usando a métrica de Minkowski.

    Parâmetros:
    points (array-like): Matriz de pontos de dados (n_samples, n_features).
    p (float): Parâmetro da métrica de Minkowski. p=1 corresponde à distância de Manhattan,
               p=2 corresponde à distância Euclidiana.

    Retorna:
    dist_matrix (ndarray): Matriz de distâncias de tamanho (n_samples, n_samples).
    """
    n_samples = points.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            dist = minkowski_dist(points[i], points[j], p)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    return dist_matrix
