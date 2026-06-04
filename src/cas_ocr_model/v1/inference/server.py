"""TCP 推理服务：接收图像字节流，<END> 标记结束，返回计算结果字符串。"""

from __future__ import annotations

import socket
import threading

import cv2
import numpy as np

from ..data_modules.device import get_recommended_device
from .predictor import load_models, predict_validate_code

_END_MARKER = b"<END>"
_DEFAULT_PORT = 21601


def _handle_pic(image_bytes: bytes, device=None, models=()) -> str:
    if device is None:
        device = get_recommended_device()
    if not models:
        models = load_models(device)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    result, expr, *_ = predict_validate_code(
        image, device, *models, print_result=True
    )
    return expr


def _handle_client(client_socket, client_address, models, device) -> None:
    print(f"[server] connection from: {client_address}")
    image_data = b""
    receive_error = False
    while True:
        try:
            data = client_socket.recv(1024)
        except Exception:  # noqa: BLE001
            receive_error = True
            break
        if not data:
            break
        image_data += data
        if image_data.endswith(_END_MARKER):
            image_data = image_data[: -len(_END_MARKER)]
            break

    if receive_error:
        print(f"[{client_address}] receive error")
        return
    print(f"[{client_address}] received")

    try:
        result = _handle_pic(image_data, device, models)
    except Exception as exc:  # noqa: BLE001
        print(f"[{client_address}] inference error: {exc}")
        result = ""
    print(result)

    try:
        client_socket.sendall(result.encode("utf-8"))
    except Exception:  # noqa: BLE001
        print(f"[{client_address}] send error")
    client_socket.close()
    print(f"[{client_address}] closed")


def start_server(host: str = "0.0.0.0", port: int = _DEFAULT_PORT) -> None:
    device = get_recommended_device()
    print("[server] loading models...")
    models = load_models(device)
    print("[server] models ready")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[server] listening on {host}:{port}")

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            thread = threading.Thread(
                target=_handle_client,
                args=(client_socket, client_address, models, device),
            )
            thread.start()
    finally:
        server_socket.close()


if __name__ == "__main__":
    start_server()
