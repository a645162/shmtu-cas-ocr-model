import torch
import platform


def get_recommended_device():
    """
    获取推荐的设备
    :return:
    """
    print("Checking device...")

    if 'Darwin' in platform.system():
        # 如果是macOS系统，则启用Metal后端
        if torch.backends.mps.is_available():
            # Apple Metal可用
            print("Metal backend is available.")
            print("Using Apple Metal Backend...")
            return torch.device("mps")
        else:
            print("Metal backend is not available!")
    else:
        if torch.cuda.is_available():
            # CUDA可用
            print("CUDA is available!")
            print("Using NVIDIA CUDA...")
            return torch.device("cuda")
        else:
            # CUDA不可用
            print("CUDA is not available!")

    print("Using CPU...")
    return torch.device("cpu")


if __name__ == '__main__':
    device = get_recommended_device()
    print("Device:", device)
