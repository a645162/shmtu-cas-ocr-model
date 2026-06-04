import os
import shutil
from cas_ocr_model.v1.utils.files.dirs import create_dirs
from cas_ocr_model.v1.utils.files.get_files import divide_files_into_processes

current_source_dir: str
current_output_dir: str


def process_file(file_path):
    base_name = os.path.basename(file_path)
    source_path = os.path.join(current_source_dir, base_name)
    if not os.path.exists(source_path):
        return
    output_path = os.path.join(current_output_dir, base_name)
    shutil.copyfile(source_path, output_path)


def copy_img_by_cluster_result(
        source_dir: str, classify_dir: str,
        output_dir: str,
):
    create_dirs([output_dir])

    global current_source_dir
    global current_output_dir
    current_source_dir = source_dir
    current_output_dir = output_dir

    divide_files_into_processes(
        classify_dir,
        10,
        process_file,
        include_subdir=False
    )


if __name__ == "__main__":
    copy_img_by_cluster_result(
        "../../workdir/OriData/ori_gray",
        "../workdir/resnet18/equal_symbol/a",
        "../workdir/Classify/OriImg/a"
    )

    copy_img_by_cluster_result(
        "../../workdir/OriData/ori_gray",
        "../workdir/resnet18/equal_symbol/b",
        "../workdir/Classify/OriImg/b"
    )
