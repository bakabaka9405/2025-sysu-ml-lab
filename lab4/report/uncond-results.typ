// 无条件生成结果展示
#import "components.typ": epoch-samples-grid

#let uncond-assets = "assets/无条件生成结果"

// 训练过程中各 epoch 的采样结果
#let uncond-epoch-samples = figure(
	epoch-samples-grid(uncond-assets),
	caption: [无条件生成模型训练过程中各 epoch 的采样结果],
) 

// 最终生成结果
#let uncond-final-samples = figure(
	image(uncond-assets + "/final_samples.png", width: 50%),
	caption: [无条件生成模型的最终采样结果],
) 
