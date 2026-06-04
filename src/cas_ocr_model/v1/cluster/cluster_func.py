import os
import shutil
from typing import List

import cv2
import numpy as np
from sklearn.cluster import KMeans


def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            images.append(img)
    return images


def preprocess_images(images):
    flattened_images = [img.flatten() for img in images]
    return np.array(flattened_images)


def apply_kmeans(images, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(images)
    return kmeans.labels_


def copy_images_to_clusters(input_directory, output_directory, labels):
    for label in set(labels):
        cluster_directory = os.path.join(output_directory, f"cluster_{label}")
        os.makedirs(cluster_directory, exist_ok=True)

    for filename, label in zip(os.listdir(input_directory), labels):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_directory, filename)
            cluster_directory = os.path.join(output_directory, f"cluster_{label}")
            shutil.copy(img_path, cluster_directory)


def cluster_images(
        input_directories: List[str],
        output_directory: str,
        num_clusters
):
    for idx, input_directory in enumerate(input_directories, start=1):
        images = load_images_from_directory(input_directory)
        preprocessed_images = preprocess_images(images)

        # Apply K-means clustering
        labels = apply_kmeans(preprocessed_images, num_clusters)

        # Copy images to corresponding clusters
        copy_images_to_clusters(input_directory, output_directory, labels)
