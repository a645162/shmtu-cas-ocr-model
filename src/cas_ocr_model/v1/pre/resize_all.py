from cas_ocr_model.v1.utils.files.dirs import create_dirs
from cas_ocr_model.v1.utils.files.get_files import divide_files_into_processes

import cv2

save_dir = "../workdir/ori_gray_div_last"
create_dirs([save_dir])


def process_file(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print("无法读取图像", file_path)
        return

    image = cv2.resize(image, (72, 72))
    cv2.imwrite(file_path, image)


if __name__ == "__main__":
    divide_files_into_processes(
        "../workdir/ori_gray_div_last",
        14,
        process_file
    )
