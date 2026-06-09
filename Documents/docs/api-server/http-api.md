# HTTP 接口

API 服务器提供以下 HTTP 端点：

## 健康检查

### GET /api/health

返回服务健康状态。

**响应示例**：

```json
{
  "status": "ok"
}
```

## 服务状态

### GET /api/status

返回服务详细状态信息。

**响应示例**：

```json
{
  "status": "ok",
  "model_kind": "v2",
  "device": "cuda",
  "workers": 4,
  "queue_size": 0,
  "uptime_seconds": 3600
}
```

## OCR 识别 (JSON)

### POST /api/ocr

接收 Base64 编码的图片数据，返回识别结果。

**请求体**：

```json
{
  "image": "<base64 编码的图片数据>"
}
```

**响应示例 (V2)**：

```json
{
  "digit_left": "9",
  "operator": "-",
  "digit_right": "2",
  "expression": "9-2",
  "equalSymbol": -1,
  "operator_int": 2
}
```

**响应示例 (V1)**：

```json
{
  "digit_left": "9",
  "operator": "-",
  "operator_int": 2,
  "digit_right": "2",
  "equalSymbol": 1,
  "expression": "9-2"
}
```

## OCR 识别 (文件上传)

### POST /api/ocr/upload

接收 multipart/form-data 上传的图片文件，返回识别结果。

**请求格式**：`multipart/form-data`，字段名为 `file`

**cURL 示例**：

```bash
curl -X POST http://localhost:21600/api/ocr/upload \
    -F "file=@captcha.jpg"
```

**Python 示例**：

```python
import requests

with open("captcha.jpg", "rb") as f:
    resp = requests.post("http://localhost:21600/api/ocr/upload", files={"file": f})
    print(resp.json())
```

**响应格式**：与 `POST /api/ocr` 相同。

## 错误响应

所有端点在出错时返回：

```json
{
  "error": "错误描述信息"
}
```

常见 HTTP 状态码：

| 状态码 | 说明 |
|---|---|
| 200 | 成功 |
| 400 | 请求格式错误 |
| 500 | 推理失败 |
| 503 | 队列已满 |
