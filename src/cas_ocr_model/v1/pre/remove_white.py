import os

from cas_ocr_model.v1.utils.files.dirs import create_dirs
from cas_ocr_model.v1.utils.files.get_files import divide_files_into_processes, get_all_files

import cv2

save_dir = "../../workdir/Spilt/ori_gray_div_last"
create_dirs([save_dir])


def is_blank(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    # Reverse the threshold image
    threshold = cv2.bitwise_not(threshold)
    return cv2.countNonZero(threshold) == 0


def process_file(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print("无法读取图像", file_path)
        return

    if is_blank(image):
        os.remove(file_path)


def remove_blank_images(path):
    print(path)
    divide_files_into_processes(
        path,
        14,
        process_file
    )

    all_files = get_all_files(
        path,
        include_subdir=False
    )
    print("Now:", len(all_files))
    print()


if __name__ == "__main__":
    # remove_blank_images("../workdir/resnet18/equal_symbol/symbol")
    # remove_blank_images("../workdir/resnet18/equal_symbol/chs")
    #
    # remove_blank_images("../workdir/resnet18/equal_symbol/a")
    # remove_blank_images("../workdir/resnet18/equal_symbol/b")

    remove_blank_images("../../workdir/Spilt/ori_gray_div_last")
