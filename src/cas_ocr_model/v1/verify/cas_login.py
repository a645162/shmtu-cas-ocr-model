import time

from selenium.webdriver.common.by import By


def try_login(
        driver,
        user_id: str = "",
        user_pwd: str = "",
        validate_code: str = ""
):
    user_id = user_id.strip()
    user_pwd = user_pwd.strip()
    validate_code = validate_code.strip()

    elem_id = driver.find_element(By.CSS_SELECTOR, "#username")
    # elem_id = driver.find_element(By.ID, "username")
    elem_id.send_keys(user_id)
    elem_pwd = driver.find_element(By.ID, "password")
    elem_pwd.send_keys(user_pwd)

    if len(validate_code) > 0:
        elem_verify_code = driver.find_element(By.ID, "validateCode")
        elem_verify_code.send_keys(validate_code)

    elem_submit = driver.find_element(By.CSS_SELECTOR, "#login-form-controls > div > button")
    elem_submit.click()


def get_login_error_text(driver):
    elem_error = driver.find_element(By.CSS_SELECTOR, "#loginErrorsPanel")
    return str(elem_error.text).strip()


def check_validate_code_is_correct_by_error_code(text: str):
    if "用户名或密码" in text:
        return True
    return False
