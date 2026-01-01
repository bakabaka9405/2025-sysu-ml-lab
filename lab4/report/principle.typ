= 实验原理

== 前向扩散过程

扩散模型的核心思想源于非平衡热力学，通过定义一个逐步向数据添加噪声的马尔科夫链来破坏数据结构，再学习其逆过程以实现数据生成。给定初始数据分布 $q(x_0)$，前向扩散过程定义为一系列条件高斯转移：

$ q(x_t | x_(t-1)) = cal(N)(x_t; sqrt(1-beta_t) x_(t-1), beta_t bold(I)) $

其中 $beta_t in (0,1)$ 为预定义的噪声调度参数，控制每一步添加噪声的强度。通过重参数化技巧，可以直接从 $x_0$ 采样任意时刻的 $x_t$：

$ x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon, quad epsilon tilde cal(N)(0, bold(I)) $

此处 $alpha_t = 1 - beta_t$，累积乘积 $overline(alpha)_t = product_(s=1)^t alpha_s$ 表示原始信号在时刻 $t$ 的保留比例。当 $t$ 足够大时，$overline(alpha)_T approx 0$，$x_T$ 近似服从标准高斯分布。

噪声调度的选取对模型性能有显著影响。线性调度将 $beta_t$ 从 $beta_1 = 10^(-4)$ 线性增长至 $beta_T = 0.02$，这一设计使得前期噪声添加较为缓慢，后期加速破坏图像结构。余弦调度则通过余弦函数定义 $overline(alpha)_t$ 的衰减曲线：

$ overline(alpha)_t = (cos((t\/T + s) / (1+s) dot pi / 2))^2 \/ (cos(s / (1+s) dot pi / 2))^2 $

其中偏移量 $s = 0.008$ 用于避免 $t=0$ 时 $beta_t$ 过小。余弦调度使得信号衰减更为平滑，通常能够获得更好的生成质量。

== 反向采样过程

生成过程需要学习前向扩散的逆过程。根据贝叶斯定理，后验分布 $q(x_(t-1)|x_t, x_0)$ 同样服从高斯分布：

$ q(x_(t-1)|x_t, x_0) = cal(N)(x_(t-1); tilde(mu)_t(x_t, x_0), tilde(beta)_t bold(I)) $

其中后验均值与方差分别为：

$ tilde(mu)_t(x_t, x_0) = (sqrt(overline(alpha)_(t-1)) beta_t) / (1-overline(alpha)_t) x_0 + (sqrt(alpha_t)(1-overline(alpha)_(t-1))) / (1-overline(alpha)_t) x_t $

$ tilde(beta)_t = (1-overline(alpha)_(t-1)) / (1-overline(alpha)_t) beta_t $

由于生成时 $x_0$ 未知，模型需要对其进行估计。DDPM 采用噪声预测网络 $epsilon_theta(x_t, t)$ 预测添加到 $x_0$ 上的噪声，进而通过下式重构 $x_0$：

$ hat(x)_0 = (x_t - sqrt(1-overline(alpha)_t) epsilon_theta(x_t, t)) / sqrt(overline(alpha)_t) $

将估计的 $hat(x)_0$ 代入后验均值公式即可完成单步采样。完整的采样过程从 $x_T tilde cal(N)(0, bold(I))$ 开始，逐步执行 $T$ 次去噪操作得到最终生成结果 $x_0$。

== 训练目标

DDPM 的训练目标源于变分下界的推导。通过对数似然的分解与 KL 散度的简化，可以证明优化变分下界等价于最小化噪声预测误差：

$ cal(L)_"simple" = E_(x_0, epsilon, t) [||epsilon - epsilon_theta(sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon, t)||^2] $

训练时随机采样时间步 $t tilde "Uniform"({1,...,T})$，噪声 $epsilon tilde cal(N)(0, bold(I))$，通过前向过程生成含噪样本 $x_t$，网络预测噪声并计算均方误差损失。这一简化目标去除了 ELBO 中的时间步权重系数，实验表明能够生成更高质量的样本。

== U-Net 网络架构

噪声预测网络采用 U-Net 架构，其编码器-解码器结构配合跳跃连接能够同时捕获图像的局部细节与全局语义。网络的核心组件包含残差块与时间嵌入注入模块。

时间步编码将离散的时间步 $t$ 映射为高维向量。正弦位置编码借鉴 Transformer 的设计：

$ "PE"(t)_(2i) = sin(t / 10000^(2i/d)), quad "PE"(t)_(2i+1) = cos(t / 10000^(2i/d)) $

可学习位置编码则通过嵌入层直接学习每个时间步对应的向量表示。时间嵌入经过多层感知机变换后，以加法方式注入到残差块的中间特征中，使网络能够感知当前去噪阶段。

残差块由两个卷积-归一化-激活序列组成，其间注入时间嵌入。编码器通过卷积下采样逐步提取高层特征，解码器通过转置卷积上采样恢复空间分辨率，跳跃连接将编码器各层特征与解码器对应层拼接以保留细节信息。

== DDIM 加速采样

标准 DDPM 需要 $T=1000$ 步采样，计算开销较大。DDIM 通过定义非马尔科夫扩散过程实现加速采样。给定子序列时间步 $tau_1 < tau_2 < ... < tau_S$，DDIM 的更新公式为：

$ x_(tau_(i-1)) = sqrt(overline(alpha)_(tau_(i-1))) hat(x)_0 + sqrt(1-overline(alpha)_(tau_(i-1)) - sigma^2) epsilon_theta(x_(tau_i), tau_i) + sigma epsilon $

其中 $sigma$ 控制采样的随机性。当 $sigma = 0$ 时为确定性采样，给定相同初始噪声将生成相同结果；当 $sigma = sqrt((1-overline(alpha)_(t-1))/(1-overline(alpha)_t)) sqrt(1-overline(alpha)_t\/overline(alpha)_(t-1))$ 时退化为标准 DDPM。DDIM 可以在 50 步甚至更少的步数下获得与 1000 步 DDPM 相当的生成质量。

== 条件生成

条件生成通过引入类别标签 $y$ 控制生成内容。将标签通过嵌入层映射为向量后与时间嵌入相加，即可使网络同时感知类别与时间步信息，学习条件分布 $p_theta(x_(t-1)|x_t, y)$。采样时只需将目标类别的嵌入向量注入网络，即可生成对应类别的样本。

== 标签嵌入插值

条件生成模型的标签嵌入层将离散的类别索引映射至连续的向量空间，这一表示空间具有语义平滑性，即相近的嵌入向量对应相似的生成结果。基于此性质，可以通过在嵌入空间中进行插值来实现类别间的平滑过渡。

给定起始类别 $y_"start"$ 与目标类别 $y_"end"$，设其对应的嵌入向量分别为 $e_"start" = E(y_"start")$ 与 $e_"end" = E(y_"end")$，其中 $E$ 为嵌入层。线性插值嵌入定义为：

$ e_alpha = (1-alpha) dot e_"start" + alpha dot e_"end", quad alpha in [0,1] $

当 $alpha = 0$ 时插值嵌入等于起始类别嵌入，生成结果应为起始数字；当 $alpha = 1$ 时等于目标类别嵌入，生成结果应为目标数字；中间取值的插值嵌入将产生兼具两类特征的过渡形态。采样时将插值嵌入 $e_alpha$ 直接替代标签嵌入注入网络，保持其余采样过程不变。为确保生成序列的连贯性，同一组插值实验使用相同的初始噪声。

== 无分类器引导

无分类器引导是一种提升条件生成质量的技术，通过在采样阶段融合有条件与无条件预测来增强生成结果与目标类别的一致性。传统分类器引导方法需要额外训练噪声分类器，而无分类器引导仅需对原有扩散模型进行简单修改即可实现。

训练阶段采用条件丢弃策略，以一定概率 $p_"drop"$ 将输入标签置为空值，使模型同时学习条件分布 $epsilon_theta(x_t, t, y)$ 与无条件分布 $epsilon_theta(x_t, t, emptyset)$。采样阶段通过引导系数 $s$ 对两种预测进行线性组合：

$ tilde(epsilon)_theta(x_t, t, y) = epsilon_theta(x_t, t, emptyset) + s dot (epsilon_theta(x_t, t, y) - epsilon_theta(x_t, t, emptyset)) $

当 $s = 1$ 时退化为标准条件采样，$s > 1$ 时增强条件信号使生成结果更贴近目标类别。较大的引导系数能够显著提升生成样本与条件的匹配度，但过高的系数可能导致样本多样性下降或出现过饱和现象。

== 图像修复

扩散模型可自然扩展至图像修复任务。给定部分遮挡的图像 $x_"known"$ 与二值掩码 $m$，修复过程在每个去噪步骤后强制保持已知区域不变：

$ x'_(t-1) = m dot.o q(x_(t-1)|x_"known") + (1-m) dot.o p_theta(x_(t-1)|x_t) $

其中 $q(x_(t-1)|x_"known")$ 通过前向过程将原始已知区域加噪至对应时间步。这一策略使模型在生成缺失区域时能够参考已知区域的上下文信息，实现语义连贯的补全。
