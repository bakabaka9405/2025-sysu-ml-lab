= 实验结果

#import "uncond-results.typ": uncond-epoch-samples, uncond-final-samples

== 无条件生成结果

无条件模型在训练过程中各轮次的采样结果如下图所示。训练初期生成的样本噪声较多且结构模糊，随着训练的推进，生成质量逐步提升。中后期的样本已能呈现清晰的数字轮廓与自然的笔画细节。

#uncond-epoch-samples

训练完成后的最终生成结果展示了模型的生成能力。模型能够生成形态各异的手写数字，数字分布大致均匀覆盖 0 至 9 各类别。生成样本呈现自然的手写风格变化，笔画粗细与倾斜角度均表现出多样性。部分样本存在轻微模糊或变形，但整体可识别性良好。

#uncond-final-samples

== 去噪过程可视化

@fig:denoise 展示了无条件生成模型的去噪过程。从纯高斯噪声开始，图像经历 1000 步去噪逐渐呈现清晰的数字轮廓。采样初期主要消除高频噪声成分，图像整体仍呈现随机纹理。随着采样的推进，数字的基本形状开始显现。采样后期数字轮廓趋于稳定，后续步骤主要优化细节与边缘。由于缺乏类别引导，各样本最终生成的数字类别具有随机性。

#figure(
  image("assets/去噪可视化结果.png", width: 50%),
  caption: [无条件生成模型的去噪过程可视化],
) <fig:denoise>

== 配置组合对比

@fig:compare_config 对比了四种配置组合的生成质量。从视觉效果观察，采用线性调度的配置生成的数字边缘更加锐利且细节更为清晰，而采用余弦调度的配置生成的样本略显平滑。正弦编码与可学习编码之间的差异并不明显，两种编码方式均能生成高质量样本。综合对比结果，正弦编码配合线性调度为推荐配置。

#let compare-assets = "assets/四个无条件生成对比"

#figure(
  grid(
    columns: 2,
    gutter: 8pt,
    figure(image(compare-assets + "/sinusoidal_linear.png"), caption: [sinusoidal + linear], numbering: none),
    figure(image(compare-assets + "/sinusoidal_cosine.png"), caption: [sinusoidal + cosine], numbering: none),

    figure(image(compare-assets + "/learnable_linear.png"), caption: [learnable + linear], numbering: none),
    figure(image(compare-assets + "/learnable_cosine.png"), caption: [learnable + cosine], numbering: none),
  ),
  caption: [不同配置组合的生成质量对比],
) <fig:compare_config>

@fig:compare_loss 展示了四种配置的训练损失曲线。所有配置在 10 个轮次内均实现收敛，损失值从初始约 0.065 下降至 0.02 至 0.03 区间。线性调度相较余弦调度收敛更快且最终损失值更低，正弦编码与可学习编码的线性调度配置最终损失均稳定在 0.018 左右，而余弦调度配置的最终损失约为 0.03。正弦编码与可学习编码在相同调度下表现接近，表明两种时间步编码方式对模型拟合能力的影响有限。

#figure(
  image("assets/无条件生成Loss对比.png", width: 70%),
  caption: [四种配置组合的训练损失曲线对比],
) <fig:compare_loss>

== DDPM 与 DDIM 对比

@tbl:ddim 给出了不同采样步数下 DDIM 的性能表现。DDPM 1000 步采样耗时约 6.67 秒，而 DDIM 50 步可将时间压缩至约 0.35 秒，加速比达 19 倍。从图中可见 DDIM 10 步就已经可以在 MNIST 任务上取得较好质量，而 DDIM 100 步的生成质量反而略逊于 50 步，样本中出现轻微噪声痕迹，这可能是由于步数过多导致过拟合所致。

#figure(
  table(
    columns: 3,
    [采样方法], [步数], [采样时间],
    [DDPM], [1000], [6.67s],
    [DDIM], [100], [0.67s],
    [DDIM], [50], [0.35s],
    [DDIM], [20], [0.13s],
    [DDIM], [10], [0.08s],
  ),
  caption: [DDPM 与 DDIM 的采样时间与质量对比],
) <tbl:ddim>

#let ddim-assets = "assets/无条件DDIM对比"

@fig:ddim_compare 展示了 DDPM 与不同步数 DDIM 的采样结果对比。所有采样均使用相同的初始噪声。

#figure(
  grid(
    columns: 3,
    gutter: 8pt,
    figure(image(ddim-assets + "/ddpm_1000_steps.png"), caption: [DDPM 1000 步], numbering: none),
    figure(image(ddim-assets + "/ddim_100_steps.png"), caption: [DDIM 100 步], numbering: none),
    figure(image(ddim-assets + "/ddim_50_steps.png"), caption: [DDIM 50 步], numbering: none),
    figure(image(ddim-assets + "/ddim_20_steps.png"), caption: [DDIM 20 步], numbering: none),
    figure(image(ddim-assets + "/ddim_10_steps.png"), caption: [DDIM 10 步], numbering: none),
  ),
  caption: [DDPM 与不同步数 DDIM 的采样结果对比],
) <fig:ddim_compare>

确定性 DDIM 与随机 DDIM 的对比如@fig:ddim_eta 所示。确定性 DDIM 设置 $eta = 0$，在相同初始噪声下生成完全相同的结果，适用于需要可复现性的场景。随机 DDIM 设置 $eta = 1$，通过引入噪声增加了生成多样性，同时莫名降低了采样噪点，原因不详。

#figure(
  grid(
    columns: 2,
    gutter: 16pt,
    figure(image(ddim-assets + "/ddim_50_steps.png"), caption: [DDIM 50 步 $eta = 0$], numbering: none),
    figure(image(ddim-assets + "/ddim_50_steps_eta1.png"), caption: [DDIM 50 步 $eta = 1$], numbering: none),
  ),
  caption: [确定性 DDIM 与随机 DDIM 的对比],
) <fig:ddim_eta>

== 条件生成结果

条件生成模型能够根据指定标签生成对应数字。@fig:cond 展示了每个数字类别各生成 8 个样本的结果，样本按行排列，每行对应一个数字类别。模型准确遵循条件约束，各行样本均为对应数字，同时保持类内多样性。

#figure(
  image("assets/条件生成结果.png", width: 50%),
  caption: [条件生成模型的采样结果],
) <fig:cond>

== 无分类器引导效果

@fig:cfg 对比了有无无分类器引导的条件生成结果。两组样本使用相同的初始噪声，仅在采样阶段应用不同的引导系数。无引导组设置 `cfg_scale = 1.0`，引导组设置 `cfg_scale = 3.0`。

#let cfg-assets = "assets/CFG对比"

#figure(
  grid(
    columns: 2,
    gutter: 16pt,
    figure(image(cfg-assets + "/no_cfg.png"), caption: [无引导], numbering: none),
    figure(image(cfg-assets + "/with_cfg.png"), caption: [有引导], numbering: none),
  ),
  caption: [无分类器引导效果对比],
) <fig:cfg>

对比结果显示，引入无分类器引导后生成的数字轮廓更加清晰锐利，笔画的连贯性与完整性均有所提升。引导机制增强了条件信号的作用，使生成样本更接近标准手写体形态。具有相似结构的类别在引导后更易区分。

== 标签插值与图像修复结果

@fig:interp 展示了标签嵌入空间的插值结果。每行对应一组插值序列，从左侧起始数字平滑过渡至右侧目标数字。中间位置的样本呈现两个数字的混合特征，笔画结构逐渐从起始形态演变为目标形态。插值过程的连续性验证了嵌入空间的语义平滑性，模型学习到的类别表示具有良好的几何结构。

#figure(
  image("assets/插值结果.png", width: 70%),
  caption: [标签嵌入空间的插值结果],
) <fig:interp>

@fig:inpaint 展示了三种遮挡模式下的修复效果。中心遮挡模式下，模型能够根据保留的边缘信息合理补全中央区域，数字结构保持完整。右半遮挡模式具有更大挑战性，模型需要从左半部分推断右半部分的内容，修复结果在大多数情况下符合原始数字类别。随机遮挡模式下，模型利用分散的已知像素进行补全，重建效果依赖于保留信息的分布。

#let inpaint-assets = "assets/修复结果"

#figure(
  grid(
    columns: 3,
    gutter: 8pt,
    figure(image(inpaint-assets + "/center.png"), caption: [中心遮挡], numbering: none),
    figure(image(inpaint-assets + "/right_half.png"), caption: [右半遮挡], numbering: none),
    figure(image(inpaint-assets + "/random.png"), caption: [随机遮挡], numbering: none),
  ),
  caption: [不同遮挡模式的图像修复结果],
) <fig:inpaint>

修复结果表明扩散模型在生成过程中能够利用上下文信息，即使在大面积遮挡的情况下仍能产生语义合理的补全。修复质量受遮挡比例与位置影响，边缘信息的保留对修复效果尤为重要。

== 结论

本实验实现了基于 DDPM 的扩散模型，验证了其在 MNIST 数据集上的生成能力。实验表明 DDIM 可在保持生成质量的前提下将采样速度提升近 20 倍，CFG能有效增强条件生成的可控性，标签插值与图像修复实验进一步展示了扩散模型的应用潜力。
