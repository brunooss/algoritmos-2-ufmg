import numpy as np
import time


def is_feasible(m_distances, k, r):
    n = m_distances.shape[0]
    centers = [0]
    for i in range(1, n):
        if all(m_distances[i][c] > r for c in centers):
            centers.append(i)
            if len(centers) > k:
                return False
    return True


def assign_labels(m_distances, centers):
    n = m_distances.shape[0]
    labels = [-1] * n
    for i in range(n):
        nearest_center = min(centers, key=lambda c: m_distances[i][c])
        labels[i] = centers.index(nearest_center)
    return labels


def k_center_variable_refinement(k, m_distances, num_refinements):
    start_time = time.time()

    n = m_distances.shape[0]
    lower = 0
    upper = np.max(m_distances)

    for _ in range(num_refinements):
        mid = (lower + upper) / 2
        if is_feasible(m_distances, k, mid):
            upper = mid
        else:
            lower = mid

    end_time = time.time()
    execution_time = end_time - start_time

    centers = [0]
    for i in range(1, n):
        if all(m_distances[i][c] > upper for c in centers):
            centers.append(i)
            if len(centers) == k:
                break

    labels = assign_labels(m_distances, centers)
    return upper, labels, execution_time
