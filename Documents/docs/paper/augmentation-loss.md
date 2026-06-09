# 4. 数据增强与损失函数

## 4.1 数据增强

针对验证码图像的形态特征，本文设计了以下数据增强策略。所有增强操作均在二值化后的图像上进行，增强后再次二值化以保持纯黑白特征。

### 4.1.1 平移增强 (Translation)

以概率 $p_t = 0.7$ 对图像进行小幅平移，水平方向最大位移 6 像素，垂直方向最大位移 3 像素，不进行裁切：

$$
\mathbf{I}' = \text{WarpAffine}(\mathbf{I}, \mathbf{M}_{trans}(t_x, t_y))
$$

其中 $t_x \sim \mathcal{U}(-6, 6)$, $t_y \sim \mathcal{U}(-3, 3)$。

### 4.1.2 仿射变换 (Affine)

以概率 $p_a = 0.4$ 对图像施加轻微的旋转、剪切和缩放变换：

$$
\mathbf{I}' = \text{WarpAffine}(\mathbf{I}, \mathbf{M}_{affine})
$$

| 变换 | 参数范围 |
|---|---|
| 旋转角度 | $\theta \sim \mathcal{U}(-2.5°, 2.5°)$ |
| 剪切角度 | $\phi \sim \mathcal{U}(-4.0°, 4.0°)$ |
| 缩放因子 | $s \sim \mathcal{U}(0.97, 1.03)$ |

仿射变换矩阵的组合顺序为：平移至中心 → 剪切 → 旋转 → 缩放 → 平移回原位。使用最近邻插值 (`INTER_NEAREST`)，边界填充黑色像素。

### 4.1.3 形态学操作 (Morphology)

以概率 $p_m = 0.15$ 对图像施加随机腐蚀或膨胀：

$$
\mathbf{I}' = \begin{cases} \text{Erode}(\mathbf{I}, \mathbf{K}_{3 \times 3}), & \text{w.p. } 0.5 \\ \text{Dilate}(\mathbf{I}, \mathbf{K}_{3 \times 3}), & \text{w.p. } 0.5 \end{cases}
$$

该操作模拟验证码字符笔画的粗细变化。

### 4.1.4 稀疏噪点 (Sparse Noise)

以概率 $p_n = 0.10$ 在图像中添加稀疏椒盐噪点：

$$
\text{density} = 0.001, \quad \text{即每 1000 像素约 1 个噪点}
$$

噪点像素等概率地被置为黑 (0) 或白 (255)。

### 4.1.5 二值化参数扰动 (Binarize Jitter)

这是本文针对验证码识别任务设计的领域特定增强策略。以概率 $p_b = 0.15$ 在训练期间随机扰动二值化参数：

| 扰动项 | 范围 |
|---|---|
| 阈值抖动 (fixed 模式) | $\Delta T \sim \mathcal{U}(-12, 12)$ |
| C 参数抖动 (adaptive 模式) | $\Delta C \sim \mathcal{U}(-3, 3)$ |
| 二值化模式混入 | 从 `[min_channel_otsu, gray_otsu]` 中随机选取 |

该策略的动机是：不同的二值化参数会产生不同质量的二值图（字符断裂、粘连等），通过扰动二值化参数可增强模型对预处理质量波动的鲁棒性。

### 4.1.6 增强后重二值化

所有增强操作完成后，若 `rethreshold_after_aug = true`（默认），则对增强后的图像重新执行二值化：

$$
\mathbf{I}_{final} = \begin{cases} 255, & \mathbf{I}' > 127 \\ 0, & \text{otherwise} \end{cases}
$$

该步骤确保增强后的图像仍为纯二值图，保持训练与推理时的数据分布一致性。

## 4.2 损失函数

### 4.2.1 分类损失

三个分类头均采用带标签平滑的交叉熵损失。对于第 $k$ 个分类头：

$$
\mathcal{L}_{cls}^{(k)} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C_k} q_{i,c}^{(k)} \log p_{i,c}^{(k)}
$$

其中 $p_{i,c}^{(k)}$ 为 softmax 输出概率，$q_{i,c}^{(k)}$ 为标签平滑后的目标分布：

$$
q_{i,c}^{(k)} = \begin{cases} 1 - \epsilon + \frac{\epsilon}{C_k}, & c = y_i^{(k)} \\ \frac{\epsilon}{C_k}, & c \neq y_i^{(k)} \end{cases}
$$

标签平滑系数 $\epsilon = 0.05$。

此外，运算符分类头支持 **类别权重** 和 **Focal Loss**。类别权重 $\mathbf{w}_{op} = [1.0, 1.05, 1.1]$ 轻微平衡运算符类别不均；Focal Loss ($\gamma > 0$ 时启用) 对难样本赋予更高权重：

$$
\mathcal{L}_{focal}^{(k)} = (1 - p_{i,y_i}^{(k)})^\gamma \cdot \mathcal{L}_{ce}^{(k)}
$$

### 4.2.2 槽位顺序约束 (Slot Order Loss)

引导 3 个槽位的注意力中心从左到右依次排列：

$$
\mathcal{L}_{order} = \frac{1}{N} \sum_{i=1}^{N} \left[ \text{ReLU}(m - (c_{i,1} - c_{i,0})) + \text{ReLU}(m - (c_{i,2} - c_{i,1})) \right]
$$

其中 $c_{i,j}$ 为第 $i$ 个样本第 $j$ 个槽位的注意力中心，$m = 0.10$ 为最小间距约束。

### 4.2.3 槽位重叠惩罚 (Slot Overlap Loss)

防止不同槽位关注相同的图像区域，鼓励注意力分布正交：

$$
\mathcal{L}_{overlap} = \frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{A}_i \mathbf{A}_i^\top - \mathbf{I} \right\|_F^2
$$

其中 $\mathbf{A}_i \in \mathbb{R}^{3 \times W'}$ 为第 $i$ 个样本的注意力权重矩阵（按行归一化后），对角线元素不参与惩罚。

### 4.2.4 右边界约束 (Slot Right Boundary Loss)

约束第 3 个槽位（右数字）的注意力中心不超过等号位置。由于验证码等号通常位于图像约 70% 宽度处：

$$
\mathcal{L}_{boundary} = \frac{1}{N} \sum_{i=1}^{N} \text{ReLU}(c_{i,2} - b_{max})
$$

其中 $b_{max} = 0.68$。

### 4.2.5 注意力方差约束 (Slot Attention Variance Loss)

约束每个槽位的注意力分布不过于分散，鼓励聚焦：

$$
\mathcal{L}_{var} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=0}^{2} \text{ReLU}(\sigma_{i,j}^2 - \sigma_{max}^2)
$$

其中 $\sigma_{i,j}^2 = \sum_{l} A_{i,j,l} (p_l - c_{i,j})^2$ 为第 $j$ 个槽位注意力的方差，$\sigma_{max}^2 = 0.035$ 为方差上界。

### 4.2.6 总损失

$$
\mathcal{L} = \lambda_{dl} \mathcal{L}_{dl} + \lambda_{op} \mathcal{L}_{op} + \lambda_{dr} \mathcal{L}_{dr} + \lambda_{order} \mathcal{L}_{order} + \lambda_{overlap} \mathcal{L}_{overlap} + \lambda_{boundary} \mathcal{L}_{boundary} + \lambda_{var} \mathcal{L}_{var}
$$

默认权重配置：

| 损失项 | 权重 | 说明 |
|---|---|---|
| $\mathcal{L}_{dl}$ (左数字) | 1.0 | 分类主损失 |
| $\mathcal{L}_{op}$ (运算符) | 1.0 | 分类主损失 |
| $\mathcal{L}_{dr}$ (右数字) | 1.0 | 分类主损失 |
| $\mathcal{L}_{order}$ | 0.1 | 槽位顺序约束 |
| $\mathcal{L}_{overlap}$ | 0.05 | 槽位重叠惩罚 |
| $\mathcal{L}_{boundary}$ | 0.01 | 右边界约束 (温和启用) |
| $\mathcal{L}_{var}$ | 0.005 | 注意力方差约束 (温和启用) |
