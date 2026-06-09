---
layout: home

hero:
  name: shmtu-cas-ocr-model
  text: CAS 验证码 OCR 模型训练
  tagline: 上海海事大学统一认证平台验证码识别 — 从数据采集到模型部署
  actions:
    - theme: brand
      text: V2 快速开始
      link: /usage/v2-quickstart
    - theme: alt
      text: 论文
      link: /paper/abstract
    - theme: alt
      text: API 服务器
      link: /api-server/overview

features:
  - title: V2 — TriSlot Decoder
    details: 单 CNN 端到端识别，共享 backbone + 槽位注意力解码器，一次前向输出数字/运算符/数字三个分类头
  - title: V1 — 三模型分离
    details: 图像切割 → 独立 ResNet 分类，历史版本代码完整保留
  - title: 全流程工具链
    details: 数据采集、训练、评估、ONNX/NCNN 导出、API 服务器一键部署
  - title: 多后端推理
    details: PyTorch / ONNX Runtime / NCNN 三后端，支持 CPU、Vulkan GPU
---
