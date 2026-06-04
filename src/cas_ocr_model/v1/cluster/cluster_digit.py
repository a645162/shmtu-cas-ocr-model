from cas_ocr_model.v1.cluster.cluster_func import cluster_images

if __name__ == "__main__":
    # cluster_images(
    #     ["../workdir/Spilt/MainBody_chs/0"],
    #     "../workdir/Cluster/Digit/chs0",
    #     10
    # )

    cluster_images(
        ["../workdir/Spilt/MainBody_symbol/0"],
        "../workdir/Cluster/Digit/symbol0",
        10
    )