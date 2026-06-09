# 1. 引言

## 1.1 研究背景

验证码 (CAPTCHA, Completely Automated Public Turing test to tell Computers and Humans Apart) 作为一种区分人类用户与自动化程序的安全机制，广泛应用于互联网服务的身份认证流程中[1]。上海海事大学统一认证平台 (CAS, Central Authentication Service) 采用算术验证码作为登录安全校验手段，其验证码格式为「数字-运算符-数字 = 结果」，如图 1 所示。

此类验证码的设计初衷是利用算术运算的认知能力区分人类与机器。然而，随着深度学习技术在计算机视觉领域的快速发展，基于卷积神经网络 (CNN) 的图像分类方法已在各类验证码识别任务中展现出超越人类水平的性能[2]。传统的验证码安全性评估体系因此面临挑战，对验证码机制的鲁棒性进行系统性的量化评估具有重要的实践意义。

## 1.2 问题定义

上海海事大学 CAS 验证码具有以下特征：

1. **图像结构**：验证码图像包含一个算术表达式，格式为 $\text{digit}_L \ \text{op} \ \text{digit}_R = \text{result}$，其中 $\text{digit}_L, \text{digit}_R \in \{0, 1, \ldots, 9\}$，$\text{op} \in \{+, -, \times\}$。
2. **视觉干扰**：图像包含噪点、干扰线和字符粘连等干扰元素，但整体对比度较高，字符区域可辨识。
3. **字符排列**：字符沿水平方向从左到右依次排列，具有明确的空间位置关系。

验证码识别任务可形式化为一个多标签分类问题：给定验证码图像 $I \in \mathbb{R}^{H \times W}$，需要同时预测左数字标签 $y_{dl} \in \{0, \ldots, 9\}$、运算符标签 $y_{op} \in \{0, 1, 2\}$ 和右数字标签 $y_{dr} \in \{0, \ldots, 9\}$。

## 1.3 相关工作

传统的验证码识别方法通常采用「图像切割 + 独立分类」的流水线策略[3]。该方法首先对验证码图像进行预处理（灰度化、二值化、去噪），然后根据字符间距将图像切割为独立片段，最后将每个片段送入分类器进行识别。该方法的主要缺陷在于：

- **切割误差累积**：字符粘连或间距不均匀时，切割位置偏差将直接导致后续分类错误。
- **模型管理复杂**：需要独立训练和维护多个分类器，部署和维护成本较高。
- **信息损失**：切割操作破坏了字符间的空间关系，无法利用全局上下文信息。

近年来，基于注意力机制的端到端方法在场景文字识别 (STR) 领域取得了显著进展[4, 5]。然而，这些方法通常针对变长文本序列设计，对于验证码这种固定结构（三字符、已知位置）的场景存在过度设计的问题。

## 1.4 本文贡献

本文的主要贡献如下：

1. 提出 **TriSlot Decoder** 架构，利用可学习的槽位查询和多头注意力机制实现验证码字符的端到端识别，无需图像切割。
2. 设计了包含槽位顺序约束、重叠惩罚、右边界约束和注意力方差约束的 **结构化损失函数**，显式引导模型学习验证码的空间排列先验。
3. 构建了基于 CAS 验证码接口的 **自动化数据采集管线**，并公开了标注数据集。
4. 针对验证码图像的形态特征，设计了 **二值化参数扰动** 等领域特定的数据增强策略。
5. 实验验证了所提方法在保证高精度的同时，显著降低了部署复杂度，支持 ONNX/NCNN 轻量化推理。

## 参考文献

[1] von Ahn L, Maurer B, McMillen C, et al. reCAPTCHA: Human-based character recognition via Web security applications. Science, 2008, 321(5895): 1465-1468.

[2] Goodfellow I J, Bulatov Y, Ibarz J, et al. Multi-digit number recognition from street view imagery using deep convolutional neural networks. arXiv preprint arXiv:1312.6082, 2013.

[3] Gao H, Tang M, Liu Y, et al. Research on CAPTCHA recognition based on CNN. Journal of Computer Research and Development, 2017.

[4] Cheng Z, Bai F, Xu Y, et al. Focusing attention: Towards accurate text recognition in natural images. ICCV, 2017: 5070-5078.

[5] Li H, Wang P, Shen C. Towards end-to-end text spotting in natural scenes. ECCV, 2018: 236-252.
