import os
import cv2

from cas_ocr_model.v1.utils.files.get_files import divide_files_into_processes

dir_a = "../../workdir/OriData/ori"
dir_b = "../../workdir/OriData/ori_gray"

if not os.path.exists(dir_a):
    exit(1)

if not os.path.exists(dir_b):
    os.makedirs(dir_b)


def convert_to_grayscale(input_path: str, output_path: str):
    if not os.path.isfile(input_path):
        return
    if not (
            input_path.endswith(".png") or input_path.endswith(".jpg") or
            input_path.endswith(".jpeg") or input_path.endswith(".bmp")
    ):
        return
    try:
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: {input_path}")
            return

        # To Binary
        image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]

        cv2.imwrite(output_path, image)
    except:
        print(f"Error: {input_path}")
        return


def process_file(file_path):
    file_name = os.path.basename(file_path)
    output_path = os.path.join(dir_b, file_name)
    convert_to_grayscale(file_path, output_path)

if __name__ == "__main__":
    print("1. Converting to grayscale")
    print("2. Converting to binary")

    num_processes = 12
    print("Number Of Processes:", num_processes)

    # 调用函数分进程处理文件
    divide_files_into_processes(
        dir_a,
        num_processes,
        process_file,
        include_subdir=True
    )
