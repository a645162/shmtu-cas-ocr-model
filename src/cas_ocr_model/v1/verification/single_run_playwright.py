"""单次跑：Playwright 截图验证码并保存到本地查看。"""

from __future__ import annotations

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

CAS_LOGIN_URL = "https://cas.shmtu.edu.cn/cas/login"


def run_single() -> None:
    with sync_playwright() as p:
        browser = p.firefox.launch()
        page = browser.new_page()
        page.goto(CAS_LOGIN_URL)
        page.wait_for_selector('//*[@id="captchaImg"]')
        captcha_img = page.query_selector('//*[@id="captchaImg"]')
        screenshot_bytes = captcha_img.screenshot()
        captcha_image = cv2.imdecode(
            np.frombuffer(screenshot_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        print("captcha shape:", captcha_image.shape)
        cv2.imshow("Captcha Screenshot", captcha_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        browser.close()


if __name__ == "__main__":
    run_single()
