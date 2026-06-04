from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import cv2
import numpy as np
from selenium import webdriver
import base64

from cas_ocr_model.v1.verify.cas_login import try_login, get_login_error_text, check_validate_code_is_correct_by_error_code

option = webdriver.ChromeOptions()
# option.add_argument('headless')

driver = \
    webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()),
        options=option
    )

# 访问URL
url = "https://cas.shmtu.edu.cn/cas/login"
driver.get(url)

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

print(captcha_image.shape)

# captcha_image = cv2.resize(captcha_image, (400, 140))
# print(captcha_image.shape)

# 显示图像
cv2.imshow('Captcha Image', captcha_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

code = input("请输入验证码：").strip()

try_login(driver, "202300000000", "password", code)

error_text = get_login_error_text(driver)

# print(error_text)
print("验证码是否正确", check_validate_code_is_correct_by_error_code(error_text))

input()

# 关闭WebDriver
driver.quit()
