import cv2

from cas_ocr_model.v1.utils.files.get_files import divide_files_into_processes
from cas_ocr_model.v1.utils.files.dirs import create_dirs, get_output_dir
from cas_ocr_model.v1.utils.pic.spilt_img import spilt_img_by_ratio


def save_percent(
        image_path: str, output_path: str,
        start_ratio: float = 0.8, end_ratio: float = 1
):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像", image_path)
        return

    horizontal_part = spilt_img_by_ratio(image, start_ratio, end_ratio)

    resized_image = cv2.resize(horizontal_part, (224, 224))

    cv2.imwrite(output_path, resized_image)


save_dir = "../../workdir/Spilt/ori_gray_div_last"
create_dirs([save_dir])


def process_file(file_path):
    output_path = get_output_dir(file_path, save_dir)
    save_percent(
        file_path, output_path,
        start_ratio=src.config.config.equal_symbol_key_start,
        end_ratio=src.config.config.equal_symbol_key_end
    )


if __name__ == "__main__":
    divide_files_into_processes(
        "../workdir/OriData/ori_gray",
        10,
        process_file
    )

    # input_image_path = "../workdir/ori_gray/20240102160151_server.png"
    # output_image_path = "../workdir/20240102160151_server.png"
    #
    # save_percent(
    #     input_image_path, output_image_path,
    #     0.7
    # )
