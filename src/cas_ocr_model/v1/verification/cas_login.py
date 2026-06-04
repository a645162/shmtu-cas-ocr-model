"""Selenium 版 CAS 登录辅助（输入账号、提交、读取错误信息）。"""

from __future__ import annotations

from selenium.webdriver.common.by import By


def try_login(
    driver,
    user_id: str = "",
    user_pwd: str = "",
    validate_code: str = "",
) -> None:
    """填入用户名/密码/验证码并点击登录。"""
    user_id = user_id.strip()
    user_pwd = user_pwd.strip()
    validate_code = validate_code.strip()

    driver.find_element(By.CSS_SELECTOR, "#username").send_keys(user_id)
    driver.find_element(By.ID, "password").send_keys(user_pwd)
    if validate_code:
        driver.find_element(By.ID, "validateCode").send_keys(validate_code)
    driver.find_element(
        By.CSS_SELECTOR, "#login-form-controls > div > button"
    ).click()


def get_login_error_text(driver) -> str:
    return str(
        driver.find_element(By.CSS_SELECTOR, "#loginErrorsPanel").text
    ).strip()


def check_validate_code_is_correct_by_error_code(text: str) -> bool:
    """若错误信息含"用户名或密码"，说明验证码已通过。"""
    return "用户名或密码" in text
