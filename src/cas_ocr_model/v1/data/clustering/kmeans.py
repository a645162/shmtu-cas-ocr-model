"""KMeans 聚类核心：加载图像 → flatten → 聚类 → 按标签复制。"""

from __future__ import annotations

import os
import shutil

import cv2
import numpy as np
from sklearn.cluster import KMeans


def _load_images_from_directory(directory: str) -> list:
    images: list = []
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png")):
            images.append(cv2.imread(os.path.join(directory, filename)))
    return images


def _preprocess_images(images: list) -> np.ndarray:
    return np.array([img.flatten() for img in images])


def _apply_kmeans(image_matrix: np.ndarray, num_clusters: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(image_matrix)
    return kmeans.labels_


def _copy_images_to_clusters(
    input_directory: str, output_directory: str, labels: np.ndarray
) -> None:
    for label in set(labels):
        os.makedirs(os.path.join(output_directory, f"cluster_{label}"), exist_ok=True)
    for filename, label in zip(os.listdir(input_directory), labels, strict=False):
        if filename.endswith((".jpg", ".png")):
            src = os.path.join(input_directory, filename)
            dst = os.path.join(output_directory, f"cluster_{label}", filename)
            shutil.copy(src, dst)


def cluster_images(
    input_directories: list[str],
    output_directory: str,
    num_clusters: int,
) -> None:
    """对每个输入目录分别做 KMeans 聚类并把图像复制到 output_directory/cluster_{i}/。"""
    for input_directory in input_directories:
        images = _load_images_from_directory(input_directory)
        image_matrix = _preprocess_images(images)
        labels = _apply_kmeans(image_matrix, num_clusters)
        _copy_images_to_clusters(input_directory, output_directory, labels)
