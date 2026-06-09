# V2 数据采集

训练数据为 jpg + json 配对，json 中标注了完整的表达式和答案。

## 自动采集工具

项目内置 `cas_ocr_model.datasets.dataset_collector`，可自动化采集验证码数据：

```bash
python -m cas_ocr_model.datasets.dataset_collector \
    --backend restful \
    --ocr-url http://127.0.0.1:21600 \
    --output ./dataset \
    --count 5000 \
    --processes 4 \
    --per-process 8
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--backend` | OCR 后端: `restful` (远程 API) / `local` (本地模型) |
| `--ocr-url` | 远程 OCR 服务地址 |
| `--output` | 输出目录 |
| `--count` | 目标采集数量 |
| `--processes` | 并发进程数 |
| `--per-process` | 每进程并发数 |

## Shell 脚本采集

```bash
# 单机采集
bash scripts/collection/collect.sh

# 8 GPU 本地采集
bash scripts/collection/collect_local_8gpu.sh

# 下载已有权重
bash scripts/collection/download_weights.sh
```

## 数据集格式

每个样本由一对文件组成：

```
dataset/
  00000000.jpg        # 验证码图片
  00000000.json       # 标注信息
```

JSON 格式：

```json
{
  "expression": "9 - 2 = 7",
  "answer": "7",
  "digit_left": "9",
  "operator": "-",
  "digit_right": "2"
}
```

支持的标注格式（训练时自动解析）：

| 格式 | 示例 |
|---|---|
| 带空格 | `9 - 2 = 7` |
| 无空格 | `9-2=7` |
| 中文 | `9减2等于7` |
| 中文运算符 | `9 乘 2 = 18` |

## 数据集拆分

```bash
# 拆分训练/验证集
python scripts/data/generate_dataset_split.py --data-root ./dataset --train-ratio 0.9

# 打包数据集
python scripts/data/package_dataset_zip.py --data-root ./dataset --output ./dataset.zip
```

## 公开数据集

- **Hugging Face**: https://huggingface.co/datasets/a645162/shmtu_cas_validate_code
- **Gitee AI**: https://ai.gitee.com/datasets/a645162/shmtu_cas_validate_code
