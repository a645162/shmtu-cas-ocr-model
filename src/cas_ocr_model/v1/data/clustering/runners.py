"""聚类运行入口：数字/运算符/最后一段的常用调用。"""

from __future__ import annotations

from .kmeans import cluster_images


def cluster_digit_from_chs(chs_dir: str, output_dir: str, n_clusters: int = 10) -> None:
    """对 CHS 形式下"第一个数字"做 KMeans 聚类。"""
    cluster_images([chs_dir], output_dir, n_clusters)


def cluster_digit_from_symbol(symbol_dir: str, output_dir: str, n_clusters: int = 10) -> None:
    """对 Symbol 形式下"第一个数字"做 KMeans 聚类。"""
    cluster_images([symbol_dir], output_dir, n_clusters)


def cluster_operator(chs_dir: str, symbol_dir: str, output_dir: str, n_clusters: int = 6) -> None:
    """对运算符区域做 KMeans 聚类（CHS+Symbol 两个来源）。"""
    cluster_images([chs_dir, symbol_dir], output_dir, n_clusters)


def cluster_last_segment(segment_dir: str, output_dir: str, n_clusters: int = 3) -> None:
    """对最后一段做聚类（用于等号 vs CHS）。"""
    cluster_images([segment_dir], output_dir, n_clusters)


if __name__ == "__main__":
    # 演示：与原 cluster2.py 行为一致
    cluster_last_segment(
        "../workdir/ori_gray_div_last_classify/cluster_0",
        "../workdir/ori_gray_div_last_classify/symbol_cluster",
        n_clusters=3,
    )
