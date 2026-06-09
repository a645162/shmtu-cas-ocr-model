# TCP 协议

API 服务器同时提供 TCP 端口 (默认 21601)，适用于低延迟场景。

## 协议格式

### 请求

1. 发送图片二进制数据
2. 追加 `<END>` 标记 (ASCII 字符串)

```
[图片二进制数据]<END>
```

### 响应

返回识别的表达式字符串，以换行符结尾。

```
9-2\n
```

## Python 客户端示例

```python
import socket

def ocr_tcp(image_bytes: bytes, host: str = "127.0.0.1", port: int = 21601) -> str:
    """通过 TCP 协议发送图片并获取 OCR 结果。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(image_bytes + b"<END>")
        data = b""
        while True:
            chunk = s.recv(1024)
            if not chunk:
                break
            data += chunk
            if data.endswith(b"\n"):
                break
    return data.decode().strip()

# 使用示例
with open("captcha.jpg", "rb") as f:
    result = ocr_tcp(f.read())
print(result)  # 例: "9-2"
```

## 适用场景

| 场景 | 推荐协议 |
|---|---|
| Web 前端 / 通用 API | HTTP |
| 高吞吐低延迟 | TCP |
| 跨语言调用 | HTTP |
| 内部微服务通信 | TCP |
