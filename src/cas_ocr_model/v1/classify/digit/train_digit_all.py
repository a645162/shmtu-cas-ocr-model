import torch

from cas_ocr_model.v1.classify.utils.devices_selector import get_recommended_device
from cas_ocr_model.v1.config import config
from cas_ocr_model.v1.classify.digit.train_mnist import train_mnist
from cas_ocr_model.v1.classify.digit.train_digit import train_digit


def train_digit_all(device: torch.device):

    print("Training digit...")

    if config.pretrain_on_mnist:
        train_mnist(device)
    else:
        print("Skip pretrain on mnist")

    train_digit(device)


if __name__ == "__main__":
    device = get_recommended_device()
    train_digit_all(device)
