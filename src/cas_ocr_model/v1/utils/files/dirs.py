import os
from typing import List


def create_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_dirs(dirs: List[str]):
    for dir_name in dirs:
        create_dir(dir_name)


def get_output_dir(
        input_path: str,
        output_dir_path: str,
        add_text_front="",
        add_text_behind="",
):
    file_name = os.path.basename(input_path)

    file_name, file_ext = os.path.splitext(file_name)

    if len(add_text_front) > 0:
        file_name = add_text_front + file_name
    if len(add_text_behind) > 0:
        file_name = file_name + add_text_behind

    output_path = os.path.join(output_dir_path, file_name + file_ext)

    return output_path


if __name__ == "__main__":
    pass
