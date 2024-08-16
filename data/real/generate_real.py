import numpy as np
from ucimlrepo import fetch_ucirepo

# !!! O código abaixo demora um pouco para executar (30s-1min), devido às requisições à API do UCI ML Repository. Sugiro que leia o artigo enquanto isso! :)


def fetch_datasets():
    pen_based_recognition_of_handwritten_digits = fetch_ucirepo(id=81)
    letter_recognition = fetch_ucirepo(id=59)
    dry_bean = fetch_ucirepo(id=602)
    rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
    phishing_websites = fetch_ucirepo(id=327)
    optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)
    statlog_shuttle = fetch_ucirepo(id=148)
    htru2 = fetch_ucirepo(id=372)
    image_segmentation = fetch_ucirepo(id=50)
    banknote_authentication = fetch_ucirepo(id=267)

    datasets = [pen_based_recognition_of_handwritten_digits, letter_recognition, dry_bean, rice_cammeo_and_osmancik,
                phishing_websites, optical_recognition_of_handwritten_digits, statlog_shuttle, htru2, image_segmentation, banknote_authentication]

    n_clusters = [10, 26, 7, 2, 2, 10, 7, 2, 7, 2]

    ds = []

    for dataset in datasets:
        X = dataset.data.features
        y = dataset.data.targets
        ds.append((X.values, y.values.flatten()))

    return ds, n_clusters


fetch_datasets()
