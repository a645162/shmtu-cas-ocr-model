"""单次跑：Selenium + 人工输入验证码，验证登录是否成功。"""

from __future__ import annotations

import base64

import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from .cas_login import (
    check_validate_code_is_correct_by_error_code,
    get_login_error_text,
    try_login,
)

CAS_LOGIN_URL = "https://cas.shmtu.edu.cn/cas/login"


def run_single(
    user_id: str = "202300000000",
    password: str = "password",
) -> bool:
    """打开 CAS 登录页，截图验证码，等待人工输入，验证结果。"""
    option = webdriver.ChromeOptions()
    driver = webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()), options=option
    )
    driver.get(CAS_LOGIN_URL)

    captcha_img = driver.find_element(By.XPATH, '//*[@id="captchaImg"]')
    image_data = base64.b64decode(captcha_img.screenshot_as_base64)
    image_np = np.frombuffer(image_data, np.uint8)
    captcha_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    print("captcha shape:", captcha_image.shape)
    cv2.imshow("Captcha Image", captcha_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    code = input("请输入验证码：").strip()
    try_login(driver, user_id, password, code)
    error_text = get_login_error_text(driver)
    is_correct = check_validate_code_is_correct_by_error_code(error_text)
    print("验证码是否正确:", is_correct)
    input("按 Enter 退出...")
    driver.quit()
    return is_correct


if __name__ == "__main__":
    run_single()
