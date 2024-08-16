import time


def k_center_max_distance(k, m_distances):
    start_time = time.time()

    n = m_distances.shape[0]
    centers = [0]

    while len(centers) < k:
        max_min_distance = -1
        next_center = -1
        for i in range(n):
            if i not in centers:
                min_distance = min(m_distances[i][c] for c in centers)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    next_center = i
        centers.append(next_center)

    radius = max(min(m_distances[i][c] for c in centers) for i in range(n))

    labels = [-1] * n
    for i in range(n):
        nearest_center = min(centers, key=lambda c: m_distances[i][c])
        labels[i] = centers.index(nearest_center)

    end_time = time.time()
    execution_time = end_time - start_time

    return radius, labels, execution_time
