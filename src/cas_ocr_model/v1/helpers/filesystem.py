"""文件系统工具：目录创建、文件扫描、多进程分发、时间戳。"""

from __future__ import annotations

import multiprocessing
import os
from datetime import datetime
from typing import Callable, List


def create_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_dirs(directories: List[str]) -> None:
    for d in directories:
        create_dir(d)


def get_output_dir(
    input_path: str,
    output_dir_path: str,
    add_text_front: str = "",
    add_text_behind: str = "",
) -> str:
    """根据 input_path 拼出 output_dir_path 下对应的输出路径。"""
    file_name = os.path.basename(input_path)
    file_name, file_ext = os.path.splitext(file_name)
    if add_text_front:
        file_name = add_text_front + file_name
    if add_text_behind:
        file_name = file_name + add_text_behind
    return os.path.join(output_dir_path, file_name + file_ext)


def get_all_files(directory: str, include_subdir: bool = False) -> List[str]:
    """列出目录下所有文件，可选递归。"""
    all_files: List[str] = []
    if not os.path.exists(directory):
        return all_files
    if include_subdir:
        for root, _, files in os.walk(directory):
            for f in files:
                all_files.append(os.path.join(root, f))
    else:
        all_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
    return [f.strip() for f in all_files if f.strip() and os.path.isfile(f)]


def process_files(file_list: List[str], handle_func: Callable[[str], None]) -> None:
    """用进程池并行处理一个文件列表。"""
    with multiprocessing.Pool() as pool:
        pool.map(handle_func, file_list)


def divide_files_into_processes(
    directory: str,
    process_count: int,
    handle_func: Callable[[str], None],
    include_subdir: bool = False,
) -> None:
    """
    把目录下文件均分到多个进程并行处理。
    当文件数少于进程数时自动缩减。
    """
    print(f"[filesystem] processing: {directory}")
    all_files = get_all_files(directory, include_subdir=include_subdir)
    print(f"[filesystem] total files: {len(all_files)}")
    if not all_files:
        print(f"[filesystem] no files found in: {directory}")
        return

    while len(all_files) < process_count:
        process_count //= 2
    if process_count <= 0:
        process_count = 1

    files_per_process = len(all_files) // process_count
    divided: List[List[str]] = []
    res_files = list(all_files)

    for _ in range(process_count):
        chunk = res_files[:files_per_process]
        divided.append(chunk)
        res_files = res_files[files_per_process:]
        if not res_files:
            break
    if divided:
        divided[0].extend(res_files)

    processes = []
    for chunk in divided:
        p = multiprocessing.Process(target=process_files, args=(chunk, handle_func))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def get_now_time_str() -> str:
    """返回 'YYYY_MM_DD_HH_MM_SS' 形式的时间戳。"""
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
