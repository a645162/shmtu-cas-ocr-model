"""端到端：用训练好的模型识别 CAS 验证码并验证 Selenium 登录。"""

from __future__ import annotations

import base64
import os
from time import sleep as time_sleep

import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

from ..configs.paths import work_dir_path
from ..data_modules.device import get_recommended_device
from ..helpers.filesystem import create_dir, get_now_time_str
from ..inference.predictor import load_models, predict_validate_code
from .cas_login import (
    check_validate_code_is_correct_by_error_code,
    get_login_error_text,
    try_login,
)

CAS_LOGIN_URL = "https://cas.shmtu.edu.cn/cas/login"


def run_captcha_test_loop(
    mission_count: int = 1000,
    user_id: str = "202400000000",
    password: str = "password",
    incorrect_dir: str | None = None,
) -> None:
    """循环跑 mission_count 次端到端测试，统计正确率。"""
    incorrect_dir = incorrect_dir or os.path.join(work_dir_path, "Incorrect")
    create_dir(incorrect_dir)

    device = get_recommended_device()
    print("[captcha] loading models...")
    eq_model, op_model, digit_model = load_models(device)
    print("[captcha] models ready")

    options = webdriver.ChromeOptions()
    # options.add_argument("headless")
    driver = webdriver.Firefox(
        service=ChromeService(ChromeDriverManager().install()),
        options=options,
    )
    driver.get(CAS_LOGIN_URL)

    count_total = 1
    count_correct = 0
    count_error = 0

    for i in tqdm(range(mission_count), desc=f"Total {mission_count}"):
        if i > 0:
            count_total += 1

        captcha_img = driver.find_element(By.XPATH, '//*[@id="captchaImg"]')
        image_data = base64.b64decode(captcha_img.screenshot_as_base64)
        image_np = np.frombuffer(image_data, np.uint8)
        captcha_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        calc_result, calc_str, *_ = predict_validate_code(
            captcha_image.copy(), device, eq_model, op_model, digit_model
        )
        try_login(driver, user_id, password, str(calc_result))
        error_text = get_login_error_text(driver)
        is_correct = check_validate_code_is_correct_by_error_code(error_text)

        print(f"\n[{i + 1}/{mission_count}]: {calc_str}")
        if is_correct:
            count_correct += 1
        else:
            count_error += 1
            file_name = f"incorrect_{get_now_time_str()}.jpg"
            file_path = os.path.join(incorrect_dir, file_name)
            cv2.imwrite(file_path, captcha_image)
            print(
                f"[{count_error}({(count_error / count_total * 100):.2f}%)]Error: {file_name}"
            )

        print(f"Correct Rate: {(count_correct / count_total * 100):.2f}%")
        time_sleep(3)

    time_sleep(1)
    print(
        f"Total: {count_total}, Correct: {count_correct}, "
        f"Rate: {(count_correct / count_total * 100):.2f}%"
    )
    driver.quit()


if __name__ == "__main__":
    run_captcha_test_loop(mission_count=10)
