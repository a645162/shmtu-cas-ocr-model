import os


def count_files_in_directory(root_dir):
    for dir_path, dir_names, file_names in os.walk(root_dir):
        print(f"{dir_path}: {len(file_names)}")


if __name__ == "__main__":
    print("Counting the number of files in each directory\n")

    print("Classify Directory")
    print("+++++++++++++++++++++++++++++++")
    print("Equal Symbol")
    count_files_in_directory("./workdir/Classify/EqualSymbol")
    print("+++++++++++++++++++++++++++++++")
    print("Operator")
    count_files_in_directory("./workdir/Classify/Operator")
    print("+++++++++++++++++++++++++++++++")
    print("Digit")
    count_files_in_directory("./workdir/Classify/Digit")
    print("+++++++++++++++++++++++++++++++\n")

    print("Datasets Directory")
    print("+++++++++++++++++++++++++++++++")
    print("Equal Symbol")
    count_files_in_directory("./workdir/Datasets/EqualSymbol")
    print("+++++++++++++++++++++++++++++++")
    print("Operator")
    count_files_in_directory("./workdir/Datasets/Operator")
    print("+++++++++++++++++++++++++++++++")
    print("Digit")
    count_files_in_directory("./workdir/Datasets/Digit")
    print("+++++++++++++++++++++++++++++++")
