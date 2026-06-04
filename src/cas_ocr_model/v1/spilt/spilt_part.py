import os
from typing import List

import cv2

from cas_ocr_model.v1.config import config
from cas_ocr_model.v1.utils.files.dirs import create_dirs
from cas_ocr_model.v1.utils.files.get_files import divide_files_into_processes
from cas_ocr_model.v1.utils.pic.spilt_img import spilt_img_by_ratio

output_base_dir = "../workdir/Spilt/MainBody"
current_key_point: List[float]


def spilt_img_file(file_path: str):
    base_name = os.path.basename(file_path)
    base_name_no_ext = os.path.splitext(base_name)[0]
    ext_name = os.path.splitext(base_name)[1]
    # dir_name = os.path.basename(os.path.dirname(file_path))

    image = cv2.imread(file_path)
    if image is None:
        print("无法读取图像", file_path)
        return
    # image = spilt_img(
    #     image,
    #     0,
    #     config.equal_symbol_key_start
    # )

    global current_key_point
    key_point = current_key_point.copy()

    if len(key_point) == 0:
        return

    key_point.sort()

    key_point.insert(0, 0)
    key_point.append(1)

    for i in range(len(key_point) - 1):
        now_start = key_point[i]
        now_end = key_point[i + 1]

        current_img = spilt_img_by_ratio(
            image,
            now_start,
            now_end
        )

        output_dir = os.path.join(output_base_dir, str(i))

        output_path = os.path.join(output_dir, f"{base_name_no_ext}_{i}{ext_name}")
        try:
            cv2.imwrite(output_path, current_img)
        except Exception as e:
            print("Error When Save File:", output_path)
            print(e)


def process_file(file_path: str):
    try:
        spilt_img_file(file_path)
    except Exception as e:
        pass
        print("Error When Process File:", file_path)
        print(e)


def start_spilt(
        source_dir: str,
        output_dir: str,
        key_point: List[float]
):
    global current_key_point
    global output_base_dir
    current_key_point = key_point
    output_base_dir = output_dir

    for i in range(len(current_key_point) + 1):
        output_dir = os.path.join(output_base_dir, str(i))
        create_dirs([output_dir])

    divide_files_into_processes(
        source_dir,
        10,
        process_file
    )


if __name__ == "__main__":
    if len(config.key_point_symbol) == 3:
        start_spilt(
            "../../workdir/Classify/OriImg/symbol",
            "../workdir/Spilt/MainBody_symbol",
            config.key_point_symbol
        )
        print("Spilt Key Point (Symbol) Finished!")
    else:
        print("Key Point Error(Symbol)!!!")

    if len(config.key_point_chs) == 3:
        start_spilt(
            "../../workdir/Classify/OriImg/chs",
            "../workdir/Spilt/MainBody_chs",
            config.key_point_chs
        )
        print("Spilt Key Point (CHS) Finished!")
    else:
        print("Key Point Error(CHS)!!!")
