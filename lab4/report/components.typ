// 可复用的图片网格组件

// 将图片数组排列成指定行列数的网格
// images: 图片路径数组
// cols: 列数
// row-gutter: 行间距
// col-gutter: 列间距
#let image-grid(images, cols: 5, row-gutter: 4pt, col-gutter: 4pt) = {
  let rows = calc.ceil(images.len() / cols)
  grid(
    columns: (1fr,) * cols,
    rows: auto,
    gutter: (col-gutter, row-gutter),
    ..images
      .enumerate()
      .map(((i, img)) => {
        figure(
          image(img, width: 100%),
          caption: [epoch #{ i + 1 }],
          numbering: none,
          gap: 3pt,
        )
      })
  )
}

// 生成 epoch 采样结果的 2×5 网格
// base-path: 图片基础路径（不含文件名）
// prefix: 文件名前缀，默认为 "samples_epoch_"
// count: epoch 数量，默认为 10
// suffix: 文件名后缀，默认为 ".png"
#let epoch-samples-grid(
  base-path,
  prefix: "samples_epoch_",
  count: 10,
  suffix: ".png",
) = {
  let images = range(count).map(i => base-path + "/" + prefix + str(i) + suffix)
  image-grid(images, cols: 5)
}
