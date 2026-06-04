"""基于 HuggingFace Accelerate 的 8 卡 DDP 训练项目.

一次性前向输出 3 个预测头: 第一个数字 / 运算符 / 第二个数字.

模块:
    config       - 训练/数据/模型配置 (dataclass)
    model        - CNN 主体 + 3-head 分类器 (ResNet-18 backbone)
    data         - CaptchaPairDataset: 读数据集目录的 (jpg + json) 配对
    losses       - 3-head 联合损失 (CE 加权和)
    train        - accelerate 入口: torchrun / accelerate launch 都可
    eval         - 单卡 / 多卡评估
    export       - 导出 ONNX, 供 v1 inference/predictor 替换

调用示例 (8 卡):
    accelerate launch --num_processes 8 --mixed_precision bf16 \\
        -m cas_ocr_model.trainer.train \\
        --data-root ../../../../dataset --output-dir ./runs/exp1

或 torchrun:
    torchrun --nproc_per_node=8 -m cas_ocr_model.trainer.train \\
        --data-root ../../../../dataset --output-dir ./runs/exp1
"""
