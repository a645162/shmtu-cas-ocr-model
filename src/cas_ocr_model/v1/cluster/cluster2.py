from .cluster import cluster_images

if __name__ == "__main__":
    # cluster_images(
    #     ["../workdir/ori_gray_div_last"],
    #     "output_directory",
    #     3
    # )
    cluster_images(
        ["../workdir/ori_gray_div_last_classify/cluster_0"],
        "../workdir/ori_gray_div_last_classify/symbol_cluster",
        3
    )
