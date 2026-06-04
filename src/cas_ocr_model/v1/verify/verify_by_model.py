from selenium.webdriver.common.by import By
from tqdm import tqdm
# from selenium.webdriver.firefox.service import Service as FirefoxService
# from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

import cv2
import numpy as np
from selenium import webdriver
import base64

import os
from time import sleep as time_sleep

import cv2

from cas_ocr_model.v1.utils.files.dirs import create_dir
from cas_ocr_model.v1.utils.files.get_time import get_now_time_str
from cas_ocr_model.v1.utils.pic.cv2plt import show_opencv_img_by_plt
from cas_ocr_model.v1.classify.predict.load_model import load_model, predict_cv_image
from cas_ocr_model.v1.classify.utils.devices_selector import get_recommended_device
from cas_ocr_model.v1.config import config
from cas_ocr_model.v1.classify.predict.predict_file import predict_validate_code

from cas_login import try_login, get_login_error_text, check_validate_code_is_correct_by_error_code

device = get_recommended_device()

model_equal_symbol, model_operator, model_digit = load_model(device)

test_dir_path = os.path.join(config.prj_root_path, "src", "classify", "predict", "test")

# Collection of Incorrect Questions
incorrect_dir_path = os.path.join(config.work_dir_path, "Incorrect")
create_dir(incorrect_dir_path)

options = webdriver.ChromeOptions()
# options.add_argument('headless')

driver = \
    webdriver.Firefox(
        service=ChromeService(ChromeDriverManager().install()),
        options=options
    )

# 访问URL
url = "https://cas.shmtu.edu.cn/cas/login"

mission_count = 1000

count_total = 1
count_correct = 0
count_error = 0

driver.get(url)
for i in tqdm(
        range(mission_count),
        desc=f'Total: {mission_count}'
):
    if i > 0:
        count_total += 1

    # 提取图片元素
    captcha_img = driver.find_element(By.XPATH, '//*[@id="captchaImg"]')

    # 获取图片的Base64编码数据
    captcha_img_base64 = captcha_img.screenshot_as_base64

    # 将Base64编码数据解码为图像数据
    image_data = base64.b64decode(captcha_img_base64)

    # 将图像数据转换为numpy数组
    image_np = np.frombuffer(image_data, np.uint8)

    # 解码图像
    captcha_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # print(captcha_image.shape)

    # captcha_image = cv2.resize(captcha_image, (400, 140))
    # print(captcha_image.shape)

    calc_result, calc_str, _, _, _, _ = predict_validate_code(
        captcha_image.copy(),
        device,
        model_equal_symbol,
        model_operator,
        model_digit
    )

    calc_result_str = str(calc_result)

    try_login(driver, "202400000000", "password", calc_result_str)

    error_text = get_login_error_text(driver)

    is_correct = check_validate_code_is_correct_by_error_code(error_text)
    # print(f"Is Correct: {is_correct}")

    print(f"\n[{i + 1}/{mission_count}]: {calc_str}")

    if is_correct:
        count_correct += 1
    else:
        count_error += 1

        file_name = f"incorrect_{get_now_time_str()}.jpg"
        file_path = os.path.join(incorrect_dir_path, file_name)
        cv2.imwrite(file_path, captcha_image)
        print(f"[{count_error}({(count_error / count_total * 100):.2f}%)]Error: {file_name}")

    print(f"Correct Rate: {(count_correct / count_total * 100):.2f}%")

    time_sleep(3)

time_sleep(1)
print(f"Total: {count_total}, Correct: {count_correct}, Correct Rate: {(count_correct / count_total * 100):.2f}%")

# 关闭WebDriver
driver.quit()
