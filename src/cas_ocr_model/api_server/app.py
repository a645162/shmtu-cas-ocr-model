from __future__ import annotations

import base64
import json
import signal
import socketserver
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .config import ApiServerConfig, parse_args
from .service import OcrService, QueueFullError


def _make_ocr_payload(
    *,
    success: bool,
    expression: str = "",
    result: int = 0,
    equal_symbol: int = 0,
    operator: int = 0,
    digit1: int = 0,
    digit2: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "expression": expression,
        "result": result,
        "equalSymbol": equal_symbol,
        "operator": operator,
        "digit1": digit1,
        "digit2": digit2,
        "error": error,
    }


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _parse_multipart_file(body: bytes, content_type: str) -> bytes:
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Missing multipart boundary")
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    boundary_bytes = ("--" + boundary).encode("utf-8")

    for section in body.split(boundary_bytes):
        section = section.strip()
        if not section or section == b"--":
            continue
        header_block, separator, content = section.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers = header_block.decode("utf-8", errors="ignore")
        if 'name="file"' not in headers:
            continue
        if content.endswith(b"\r\n"):
            content = content[:-2]
        if content.endswith(b"--"):
            content = content[:-2]
        return content
    raise ValueError('Multipart field "file" is required')


class OcrHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], service: OcrService) -> None:
        super().__init__(server_address, OcrHttpHandler)
        self.ocr_service = service


class OcrHttpHandler(BaseHTTPRequestHandler):
    server: OcrHttpServer

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(_compact(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Empty request body")
        if content_length > self.server.ocr_service.config.max_input_bytes * 2:
            raise ValueError("Request body too large")
        return self.rfile.read(content_length)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/health":
            self._send_json(HTTPStatus.OK, self.server.ocr_service.health_payload())
            return
        if route == "/api/status":
            self._send_json(HTTPStatus.OK, self.server.ocr_service.status_payload())
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            _make_ocr_payload(success=False, error="Not found"),
        )

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route == "/api/ocr":
                image_bytes = self._decode_json_request()
            elif route == "/api/ocr/upload":
                image_bytes = self._decode_upload_request()
            else:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    _make_ocr_payload(success=False, error="Not found"),
                )
                return

            if len(image_bytes) > self.server.ocr_service.config.max_input_bytes:
                raise ValueError("Image payload too large")

            prediction = self.server.ocr_service.submit(image_bytes)
            self._send_json(
                HTTPStatus.OK,
                _make_ocr_payload(
                    success=prediction.success,
                    expression=prediction.expression,
                    result=prediction.result,
                    equal_symbol=prediction.equal_symbol,
                    operator=prediction.operator,
                    digit1=prediction.digit1,
                    digit2=prediction.digit2,
                    error=prediction.error,
                ),
            )
        except QueueFullError as exc:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                _make_ocr_payload(success=False, error=str(exc)),
            )
        except RuntimeError as exc:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                _make_ocr_payload(success=False, error=str(exc)),
            )
        except ValueError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                _make_ocr_payload(success=False, error=str(exc)),
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                HTTPStatus.OK,
                _make_ocr_payload(success=False, error=str(exc)),
            )

    def _decode_json_request(self) -> bytes:
        body = self._read_body()
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON body") from exc
        image_base64 = payload.get("imageBase64")
        if not isinstance(image_base64, str) or not image_base64:
            raise ValueError("ImageBase64 is required")
        try:
            return base64.b64decode(image_base64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Invalid base64 string") from exc

    def _decode_upload_request(self) -> bytes:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Content-Type must be multipart/form-data")
        body = self._read_body()
        return _parse_multipart_file(body, content_type)


class OcrTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service: OcrService) -> None:
        super().__init__(server_address, OcrTcpHandler)
        self.ocr_service = service


class OcrTcpHandler(socketserver.BaseRequestHandler):
    end_marker = b"<END>"

    def handle(self) -> None:
        chunks = bytearray()
        limit = self.server.ocr_service.config.max_input_bytes + len(self.end_marker)
        while True:
            data = self.request.recv(4096)
            if not data:
                break
            chunks.extend(data)
            if self.end_marker in chunks:
                break
            if len(chunks) > limit:
                return

        marker_index = chunks.find(self.end_marker)
        if marker_index < 0:
            return
        image_bytes = bytes(chunks[:marker_index])
        if not image_bytes or len(image_bytes) > self.server.ocr_service.config.max_input_bytes:
            return

        try:
            prediction = self.server.ocr_service.submit(image_bytes)
            response = prediction.expression if prediction.success else ""
        except Exception:  # noqa: BLE001
            response = ""
        self.request.sendall(response.encode("utf-8"))


def run_servers(config: ApiServerConfig) -> int:
    service = OcrService(config)
    http_server = OcrHttpServer((config.http_host, config.http_port), service)
    tcp_server = OcrTcpServer((config.tcp_host, config.tcp_port), service)

    threads = [
        threading.Thread(target=http_server.serve_forever, name="cas-ocr-http", daemon=True),
        threading.Thread(target=tcp_server.serve_forever, name="cas-ocr-tcp", daemon=True),
    ]
    stop_event = threading.Event()

    def _stop(*_args: object) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _stop)

    for thread in threads:
        thread.start()

    status = service.status_payload()
    print(
        "[api-server] started",
        f"http={config.http_host}:{config.http_port}",
        f"tcp={config.tcp_host}:{config.tcp_port}",
        f"model_kind={config.model_kind}",
        f"device={config.device}",
        f"workers={status['poolSize']}",
        f"queue_capacity={status['queueCapacity']}",
        f"models_loaded={status['modelsLoaded']}",
    )
    if service.init_error:
        print(f"[api-server] init_error: {service.init_error}")

    try:
        while not stop_event.is_set():
            time.sleep(0.25)
    finally:
        http_server.shutdown()
        tcp_server.shutdown()
        http_server.server_close()
        tcp_server.server_close()
        service.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    return run_servers(config)
