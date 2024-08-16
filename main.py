import numpy as np
import csv
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score

from data.real.generate_real import fetch_datasets
from data.sinthetic.generate_sinthetic import generate_datasets
from k_means_approximative_1 import k_center_max_distance
from k_means_approximative_2 import k_center_variable_refinement
from k_means_np import run_k_means


def run_experiments(X, y, k, n_runs=3):
    m_distances = pairwise_distances(X)

    results = []

    for _ in range(n_runs):
        print("  a")
        radius, labels, duration = run_k_means(X, k)

        silhouette = silhouette_score(X, labels)
        rand_score = adjusted_rand_score(y, labels)
        results.append(('np', radius, silhouette, rand_score, duration))

        radius, labels, duration = k_center_max_distance(
            k, m_distances)

        silhouette = silhouette_score(X, labels)
        rand_score = adjusted_rand_score(y, labels)
        results.append(('approx1', radius, silhouette, rand_score, duration))

        radius, labels, duration = k_center_variable_refinement(
            k, m_distances, 1)
        silhouette = silhouette_score(X, labels)
        rand_score = adjusted_rand_score(y, labels)
        results.append(('approx2_5', radius, silhouette, rand_score, duration))

        radius, labels, duration = k_center_variable_refinement(
            k, m_distances, 2)
        silhouette = silhouette_score(X, labels)
        rand_score = adjusted_rand_score(y, labels)
        results.append(
            ('approx2_10', radius, silhouette, rand_score, duration))

        radius, labels, duration = k_center_variable_refinement(
            k, m_distances, 3)
        silhouette = silhouette_score(X, labels)
        rand_score = adjusted_rand_score(y, labels)
        results.append(
            ('approx2_15', radius, silhouette, rand_score, duration))

        radius, labels, duration = k_center_variable_refinement(
            k, m_distances, 4)
        silhouette = silhouette_score(X, labels)
        rand_score = adjusted_rand_score(y, labels)
        results.append(
            ('approx2_20', radius, silhouette, rand_score, duration))

        radius, labels, duration = k_center_variable_refinement(
            k, m_distances, 5)
        silhouette = silhouette_score(X, labels)
        rand_score = adjusted_rand_score(y, labels)
        results.append(
            ('approx2_25', radius, silhouette, rand_score, duration))

    return results


real_datasets, real_n_clusters = fetch_datasets()
sinthetic_datasets, sinthetic_n_clusters = generate_datasets()

datasets = real_datasets + sinthetic_datasets
n_clusters = real_n_clusters + sinthetic_n_clusters


def save_results_to_files(datasets):
    txt_file_path = "./data/generated/experiment_results.txt"
    csv_file_path = "./data/generated/experiment_results.csv"

    with open(txt_file_path, 'w') as txt_file, open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(['Dataset Index', 'Execution', 'Algorithm Type',
                            'Radius', 'Silhouette', 'Rand Index', 'Duration'])

        for dataset_index, dataset in enumerate(datasets):
            X, y = dataset
            print(dataset_index)
            k = n_clusters[dataset_index]

            print(f"Running experiments for dataset {dataset_index}...")

            results = run_experiments(X, y, k)

            for i, (algtype, radius, silhouette, rand_index, duration) in enumerate(results):
                txt_line = f"{dataset_index} {i + 1} {algtype} {radius:.4f} {silhouette:.4f} {rand_index:.4f} {duration:.4f}\n"
                txt_file.write(txt_line)

                csv_writer.writerow(
                    [dataset_index, i + 1, algtype, radius, silhouette, rand_index, duration])


save_results_to_files(datasets)
