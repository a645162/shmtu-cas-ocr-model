from cas_ocr_model.v1.cluster.cluster_func import cluster_images

if __name__ == "__main__":
    cluster_images(
        ["../workdir/Spilt/MainBody_chs/1"],
        "../workdir/Cluster/Operator/chs1",
        6
    )

    cluster_images(
        ["../workdir/Spilt/MainBody_symbol/1"],
        "../workdir/Cluster/Operator/symbol1",
        6
    )