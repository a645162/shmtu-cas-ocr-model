import os
import multiprocessing
from typing import List


def process_file_example(file_path: str):
    print(file_path)


def process_files(file_list: str, handle_func: callable(str)):
    with multiprocessing.Pool() as pool:
        pool.map(handle_func, file_list)


def get_all_files(directory: str, include_subdir: bool = False):
    all_files: List[str] = []

    if not os.path.exists(directory):
        return all_files

    if include_subdir:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    else:
        all_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]

    all_files = [
        f.strip()
        for f in all_files
        if len(f.strip()) > 0 and os.path.isfile(f)
    ]

    return all_files


def divide_files_into_processes(
        directory: str,
        process_count: int,
        handle_func: callable(str),
        include_subdir: bool = False
):
    print("Start Processing Files In:\n", directory)
    # 获取目录中的所有文件
    all_files = [
        f
        for f in get_all_files(
            directory,
            include_subdir=include_subdir
        )
    ]

    # 仅处理前 30 个文件(Debug)
    # all_files = all_files[:30]

    print("Total Files Count:", len(all_files))
    if len(all_files) <= 0:
        print("No Files Found In:", directory)
        return

    while len(all_files) < process_count:
        process_count //= 2
        # print("Reduce Process Count To:", process_count)

    if process_count <= 0:
        process_count = 1

    # 计算每个进程要处理的文件数量
    files_per_process: int = len(all_files) // process_count

    # 将文件分成 count 份，每份包含 files_per_process 个文件
    divided_files: List[List[str]] = []

    # res_files = all_files.copy()
    res_files = all_files

    for i in range(process_count):
        divided_files.append(res_files[:files_per_process])
        res_files = res_files[files_per_process:]
        if len(res_files) == 0:
            break
    if len(divided_files) > 0:
        divided_files[0].extend(res_files)

    # 创建并启动多个进程，每个进程处理一份文件列表
    processes = []
    for files_chunk in divided_files:
        process = multiprocessing.Process(
            target=process_files, args=(files_chunk, handle_func,)
        )
        process.start()
        processes.append(process)

    # 等待所有进程完成
    for process in processes:
        process.join()


if __name__ == "__main__":
    num_processes = 10
    print("Number Of Processes:", num_processes)

    # 调用函数分进程处理文件
    divide_files_into_processes(
        "../../workdir",
        num_processes,
        process_file_example,
        include_subdir=True
    )
