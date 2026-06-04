import os
import random
import shutil


def split_dataset(dataset_dir, save_dir_base, splits=None, split_ratio=None):
    if split_ratio is None:
        splits = ['train', 'val', 'test']
        split_ratio = [0.8, 0.1, 0.1]
    if splits is None:
        splits = ['train', 'val', 'test']
        split_ratio = [0.8, 0.1, 0.1]

    # 确保保存目录存在
    for split_name in splits:
        split_dir = os.path.join(save_dir_base, split_name)
        os.makedirs(split_dir, exist_ok=True)

    # 获取子目录列表
    classes = os.listdir(dataset_dir)

    # 对每个类别进行处理
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 获取类别下的所有文件
        files = os.listdir(class_dir)
        random.shuffle(files)  # 打乱文件列表顺序

        # 计算每个类别的划分数量
        num_files = len(files)
        num_splits = len(splits)
        split_counts = [int(num_files * ratio) for ratio in split_ratio]
        split_counts[-1] += num_files - sum(split_counts)

        # 随机选择文件并分配到相应的划分中
        start_index = 0
        for split_name, split_count in zip(splits, split_counts):
            split_dir: str = str(os.path.join(save_dir_base, split_name, class_name))
            os.makedirs(split_dir, exist_ok=True)
            end_index = start_index + split_count
            for file_name in files[start_index:end_index]:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(split_dir, file_name)
                shutil.copy(src, dst)
            start_index = end_index


if __name__ == '__main__':
    split_dataset(
        dataset_dir='../../workdir/Classify/EqualSymbol',
        save_dir_base='../../workdir/Datasets/EqualSymbol',
        splits=['train', 'val'],
        split_ratio=[0.9, 0.1]
    )

    split_dataset(
        dataset_dir='../../workdir/Classify/Operator',
        save_dir_base='../../workdir/Datasets/Operator',
        splits=['train', 'val'],
        split_ratio=[0.9, 0.1]
    )

    split_dataset(
        dataset_dir='../../workdir/Classify/Digit',
        save_dir_base='../../workdir/Datasets/Digit',
        splits=['train', 'val'],
        split_ratio=[0.9, 0.1]
    )
