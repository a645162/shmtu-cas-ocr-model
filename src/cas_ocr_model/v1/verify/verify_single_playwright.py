from io import BytesIO

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

url = "https://cas.shmtu.edu.cn/cas/login"

# 启动 Playwright
with sync_playwright() as p:
    browser = p.firefox.launch()

    # 创建一个新的浏览器页面
    page = browser.new_page()

    # 访问 URL
    page.goto(url)

    # 等待元素加载完成
    page.wait_for_selector('//*[@id="captchaImg"]')

    # 获取验证码图片元素
    captcha_img = page.query_selector('//*[@id="captchaImg"]')

    # 对验证码图片元素进行截图
    captcha_screenshot = captcha_img.screenshot()

    # 将截图数据转换为OpenCV格式
    captcha_image = cv2.imdecode(
        np.frombuffer(captcha_screenshot, np.uint8), cv2.IMREAD_COLOR)

    # captcha_image = cv2.resize(captcha_image, (400, 140))

    print(captcha_image.shape)

    # 显示截图
    cv2.imshow('Captcha Screenshot', captcha_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 关闭浏览器
    browser.close()
