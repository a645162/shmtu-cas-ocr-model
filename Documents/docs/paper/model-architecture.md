# 3. 模型架构

## 3.1 整体架构

本文提出的模型命名为 **CaptchaTriSlotDecoderCNN**，其整体架构如图 1 所示。模型由两部分组成：共享 CNN backbone 和 TriSlot Decoder 解码头。输入为归一化的单通道灰度图 $\mathbf{X} \in \mathbb{R}^{1 \times H \times W}$，输出为三个分类头的 logits：

$$
\hat{\mathbf{y}}_{dl}, \hat{\mathbf{y}}_{op}, \hat{\mathbf{y}}_{dr} = f(\mathbf{X}; \theta)
$$

其中 $\hat{\mathbf{y}}_{dl} \in \mathbb{R}^{10}$（左数字）、$\hat{\mathbf{y}}_{op} \in \mathbb{R}^{3}$（运算符）、$\hat{\mathbf{y}}_{dr} \in \mathbb{R}^{10}$（右数字）。

```
输入: (B, 1, 64, 192) 灰度图
         │
         ▼
┌─────────────────────┐
│   CNN Backbone      │  提取空间特征图 (B, C, H', W')
│  (MobileNetV3-Small │  ImageNet 预训练, 首层 1 通道
│   / ResNet-18 / ...)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   TriSlot Decoder   │  3 个可学习槽位查询
│                     │  多头注意力聚合
│  ┌───┐ ┌───┐ ┌───┐ │
│  │S₀ │ │S₁ │ │S₂ │ │  Slot 0 → digit_left
│  └─┬─┘ └─┬─┘ └─┬─┘ │  Slot 1 → operator
│    │     │     │    │  Slot 2 → digit_right
│    ▼     ▼     ▼    │
│  FC₁₀  FC₃   FC₁₀  │
└─────────────────────┘
         │
         ▼
输出: digit_left_logits (B,10)
      operator_logits  (B,3)
      digit_right_logits(B,10)
```

## 3.2 Backbone

Backbone 负责从输入图像中提取空间特征图。本文支持多种 backbone 架构，均适配为接收单通道灰度图输入：

| Backbone | 参数量 | 特征维度 | 特点 |
|---|---|---|---|
| MobileNetV3-Small | 2.5M | 576 | 轻量首选，推荐生产部署 |
| MobileNetV3-Large | 5.4M | 960 | 精度更高 |
| ResNet-18 | 11.7M | 512 | 经典基线 |
| ResNet-34 | 21.8M | 512 | 更强表达力 |
| ResNet-50 | 25.6M | 2048 | 大模型基线 |
| timm/* | 可变 | 可变 | 任意 timm features_only backbone |

所有 backbone 均加载 ImageNet 预训练权重，并将首层卷积从 3 通道调整为 1 通道（取预训练权重的单通道均值初始化）。

## 3.3 TriSlot Decoder

TriSlot Decoder 是本文的核心创新，其设计灵感来源于 Slot Attention[6] 机制。与场景文字识别中常用的序列解码器（如 CTC、自回归 Attention）不同，TriSlot Decoder 针对验证码三字符固定结构的特点，采用 3 个固定数量的可学习槽位查询 (Slot Query) 直接从特征图中聚合对应位置的信息。

### 3.3.1 特征投影

首先对 backbone 输出的特征图 $\mathbf{F} \in \mathbb{R}^{B \times C \times H' \times W'}$ 进行投影和位置编码：

$$
\mathbf{T} = \text{LayerNorm}(\text{Proj}(\mathbf{F}) + \text{PosConv}(\text{Proj}(\mathbf{F})))
$$

其中：
- $\text{Proj}$: 1×1 卷积 + BatchNorm + GELU，将特征维度映射到隐层维度 $d_h = 256$
- $\text{PosConv}$: 深度可分离 1D 卷积 (kernel_size=3, groups=$d_h$)，沿宽度方向编码位置信息
- 特征图沿高度维度平均池化后转置，得到 token 序列 $\mathbf{T} \in \mathbb{R}^{B \times W' \times d_h}$

### 3.3.2 槽位注意力

3 个可学习的槽位查询 $\mathbf{Q} \in \mathbb{R}^{3 \times d_h}$ 通过多头注意力机制 (Multi-Head Attention) 与特征 token 进行交互：

$$
\mathbf{S} = \text{MHA}(\mathbf{Q}, \mathbf{T}, \mathbf{T})
$$

$$
\mathbf{S} = \text{LayerNorm}(\mathbf{S} + \text{FFN}(\mathbf{S}))
$$

其中：
- 多头注意力头数 $h = 4$，每个头的维度 $d_h / h = 64$
- FFN 为两层线性变换 + GELU 激活 + Dropout
- 槽位查询通过 `nn.Parameter` 初始化，标准差 $\sigma = 0.02$

### 3.3.3 分类头

3 个槽位分别连接独立的线性分类头：

| 槽位 | 分类头 | 输出维度 | 类别含义 |
|---|---|---|---|
| Slot 0 ($\mathbf{S}_0$) | $\text{FC}_{dl}$ | 10 | 数字 0–9 |
| Slot 1 ($\mathbf{S}_1$) | $\text{FC}_{op}$ | 3 | 运算符 `+`, `-`, `*` |
| Slot 2 ($\mathbf{S}_2$) | $\text{FC}_{dr}$ | 10 | 数字 0–9 |

分类头权重初始化为正态分布 ($\sigma = 0.01$)，偏置初始化为 0。

### 3.3.4 辅助输出 (训练时)

训练时可启用 `return_aux=True`，TriSlot Decoder 额外返回：

- **slot_attention** $\mathbf{A} \in \mathbb{R}^{3 \times W'}$: 每个槽位的注意力权重分布，用于可视化和结构化损失计算。
- **slot_centers** $\mathbf{c} \in \mathbb{R}^{3}$: 每个槽位的注意力中心位置（归一化到 [0, 1]），通过加权平均计算：

$$
c_i = \sum_{j=1}^{W'} A_{i,j} \cdot \frac{j-1}{W'-1}
$$

## 3.4 与 V1 架构的对比

| 特性 | V1 (三模型分离) | V2 (TriSlot Decoder) |
|---|---|---|
| 模型结构 | 3 个独立 ResNet-18 | 1 个共享 backbone + TriSlot Decoder |
| 输入处理 | 图像切割为 3 段 | 整图输入，无需切割 |
| 运算符类别 | 6 类 (含等号) | 3 类 (`+`, `-`, `*`) |
| 参数量 | ~35M (3 × 11.7M) | ~8M (backbone + decoder) |
| 推理次数 | 3 次前向 | 1 次前向 |
| 空间建模 | 无 (切割后丢失) | 槽位注意力隐式建模 |

## 参考文献

[6] Locatello F, Weissenborn D, Unterthiner T, et al. Object-centric learning with slot attention. NeurIPS, 2020, 33: 11525-11538.
