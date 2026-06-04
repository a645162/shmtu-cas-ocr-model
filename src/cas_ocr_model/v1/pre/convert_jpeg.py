import os
import cv2

from cas_ocr_model.v1.utils.files.get_files import divide_files_into_processes


def process_file(file_path):
    image_ori = cv2.imread(file_path)
    if image_ori is None:
        return
    dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(base_name)[0]
    output_path: str = str(
        os.path.join(
            dir,
            file_name_no_ext + ".jpg"
        )
    )
    cv2.imwrite(output_path, image_ori)


if __name__ == "__main__":
    # 调用函数分进程处理文件
    divide_files_into_processes(
        "../workdir/OriData/ori_gray",
        12,
        process_file,
        include_subdir=True
    )
