from datetime import datetime


def get_now_time_str() -> str:
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    return formatted_time
