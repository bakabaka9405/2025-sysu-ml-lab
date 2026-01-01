= 实验过程

== 数据集与预处理

实验采用 MNIST 手写数字数据集，包含 60000 张训练图像与 10000 张测试图像。原始图像尺寸为 28×28 像素的灰度图，像素值范围为 0 到 255。预处理阶段首先将像素值归一化至 [0,1] 区间，随后进行 2 像素的边界填充使图像尺寸扩展为 32×32，以适配 U-Net 的下采样与上采样操作。最后将像素值线性映射至 [-1,1] 区间作为网络输入。数据加载器采用 128 的批量大小，并在每个训练轮次对数据进行随机打乱。

== 模型架构实现

U-Net 网络由编码器、中间层与解码器三部分构成。编码器包含三个分辨率层级，通道数依次为 64、128、256，每个层级包含两个残差块，相邻层级间通过步长为 2 的卷积进行下采样。中间层由两个残差块组成，用于捕获最深层的语义特征。解码器与编码器对称，通过转置卷积进行上采样，并通过跳跃连接将编码器各层特征与解码器对应层拼接后送入残差块。

残差块的实现如下所示。每个块包含两个卷积层，卷积核大小为 3×3，填充为 1 以保持空间尺寸不变。组归一化层使用 8 个分组，激活函数采用 SiLU。时间嵌入通过线性层变换后以广播加法方式注入到两个卷积层之间的特征图中。当输入输出通道数不一致时，残差连接通过 1×1 卷积进行维度匹配。

```python
class ResnetBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channel))
        self.block1 = Block(in_channel, out_channel)
        self.block2 = Block(out_channel, out_channel)
        self.res_conv = nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)
```

时间嵌入模块提供两种实现方式。正弦位置编码将时间步映射为 64 维向量，其中前 32 维为正弦分量，后 32 维为余弦分量，频率按指数衰减分布。可学习位置编码使用嵌入层将 1000 个离散时间步各自映射为 64 维向量，权重以标准差 0.1 的高斯分布初始化。两种编码方式的输出均经过两层全连接网络扩展至 256 维。

== 扩散过程实现

扩散过程的核心在于噪声调度参数的预计算。初始化阶段根据调度类型生成 $beta_t$ 序列，线性调度在 1000 步内从 $10^(-4)$ 线性增长至 0.02，余弦调度则通过余弦函数计算累积乘积 $overline(alpha)_t$ 后反推 $beta_t$。基于 $beta_t$ 可预计算训练与采样所需的全部系数并注册为模型缓冲区：

```python
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
```

前向扩散函数 `q_sample` 接收原始图像 $x_0$ 与时间步 $t$，通过预计算系数直接采样 $x_t$。训练时随机采样时间步与噪声，调用前向扩散获得含噪样本，网络预测噪声后计算均方误差损失。

反向采样函数 `p_sample` 实现单步去噪。给定当前样本 $x_t$，网络首先预测噪声 $epsilon_theta$，据此估计原始图像 $hat(x)_0$ 并裁剪至 [-1,1] 区间以提高数值稳定性。后验均值由预计算系数加权 $hat(x)_0$ 与 $x_t$ 得到，最后添加方差项完成采样。完整采样过程从标准高斯噪声开始，循环调用 1000 次单步去噪得到最终结果。

== 无条件生成训练

无条件生成模型的训练用于验证扩散模型的基本生成能力。该阶段不引入任何类别信息，模型仅学习 MNIST 数据集的整体分布特征。训练采用 AdamW 优化器，初始学习率为 $10^(-4)$。训练过程使用混合精度加速，梯度裁剪阈值设为 1.0 以防止梯度爆炸。每轮次结束后生成 16 张样本用于监控生成质量，训练过程同时记录损失曲线与梯度范数用于评估收敛状态。

== 去噪过程可视化

为直观展示扩散模型的去噪机制，实验在采样过程中记录中间状态。从纯高斯噪声开始，采样过程每隔固定步数保存一次图像状态，形成完整的演变序列。可视化结果以网格形式呈现，每行对应一个样本从噪声到清晰图像的渐进变化，展示模型如何逐步恢复图像结构。

== 配置组合对比

实验系统性地遍历时间嵌入方式与噪声调度策略的四种组合，分别为正弦编码配合线性调度、正弦编码配合余弦调度、可学习编码配合线性调度、可学习编码配合余弦调度。每种配置独立完成完整的训练流程，采用相同的优化器设置与训练轮数。训练结束后各配置分别生成样本，用于对比不同超参数组合对生成质量的影响。

== DDPM 与 DDIM 对比

DDIM 采样通过在完整时间序列中等间隔选取子序列实现加速。实验在无条件模型上设置四个采样步数分别为 10、20、50、100，对应将原始 1000 步采样压缩至相应步数。确定性 DDIM 设置 $eta=0$，随机变体设置 $eta=1$ 以对比两者差异。实验记录各配置的采样时间与生成结果，分析步数与质量的权衡关系。

DDIM 更新公式的实现如下。首先根据采样步数生成二次方时间子序列，随后按逆序遍历执行去噪。每一步从当前样本预测噪声，估计原始图像后根据公式计算下一时刻样本：

```python
pred_x0 = (img - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
pred_x0 = torch.clamp(pred_x0, -1, 1)
sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev))
dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma_t**2) * noise_pred
img = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma_t * torch.randn_like(img)
```

== 条件生成训练

条件生成模型在 U-Net 中引入标签嵌入层。嵌入层将 10 个类别索引映射为 256 维向量，对应十个数字类别 0-9。标签嵌入与时间嵌入相加后共同注入残差块，使网络能够区分不同类别的生成目标。其余训练配置与无条件模型相同，同样遍历四种编码与调度组合进行对比实验。

== 无分类器引导

无分类器引导的实现分为训练与采样两个阶段。训练阶段在损失函数计算前对标签进行随机丢弃，丢弃概率设为 0.1，被丢弃的标签以特殊值 -1 标记。网络前向传播时检测到 -1 标签的样本将不添加标签嵌入，等效于无条件预测：

```python
if y is not None and label_dropout > 0:
    drop_mask = torch.rand(y.shape[0], device=y.device) < label_dropout
    y = y.clone()
    y[drop_mask] = -1
```

采样阶段通过引导系数 `cfg_scale` 控制条件强度。当 `cfg_scale` 大于 1 时，每个去噪步骤同时计算有条件与无条件两次预测，按线性组合公式融合后用于更新样本：

```python
if cfg_scale > 1.0 and y is not None:
    y_null = torch.full_like(y, -1)
    noise_uncond = model(img, t, y_null)
    noise_cond = model(img, t, y)
    noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
```

实验设置 `cfg_scale = 3.0` 进行对比，使用 DDIM 50 步采样生成相同初始噪声下有无引导的结果。

== 标签插值与图像修复

标签插值实验验证条件嵌入空间的语义平滑性。给定两个数字类别作为起点与终点，在嵌入空间中进行线性插值生成中间状态。从嵌入层获取起点与终点类别的嵌入向量 $e_"start"$ 与 $e_"end"$，按 10 个等间隔步骤计算插值嵌入：

$ e_alpha = (1-alpha) dot e_"start" + alpha dot e_"end", quad alpha in {0, 0.11, ..., 1} $

为保证生成过程的连贯性，实验对每组插值使用相同的初始噪声。插值嵌入直接替代标签嵌入注入网络，完成采样后将结果排列为网格形式展示，每行对应一组插值序列。

图像修复实验测试扩散模型在部分遮挡条件下的补全能力。实验设计三种遮挡模式：中心遮挡将图像中央 8×8 区域置零，随机遮挡以 50% 概率保留各像素（因为大部分像素为空，实际遮挡部分不足 50%)，右半遮挡将图像右半部分置零。遮挡区域的掩码值为 0，已知区域为 1。

修复过程在每个去噪步骤后执行区域融合。已知区域通过前向扩散加噪至当前时间步，与生成区域按掩码加权合并：

```python
for i in reversed(range(0, timesteps)):
    img = diffusion.p_sample(img, t, i, y=labels)
    if i > 0:
        known_part = diffusion.q_sample(image, t_prev)
    else:
        known_part = image
    img = mask * known_part + (1 - mask) * img
```

这一操作确保已知区域信息在整个采样过程中保持不变，从而提升修复质量。实验结果以网格形式展示原始图像、遮挡图像与修复结果，直观对比模型的补全效果。
