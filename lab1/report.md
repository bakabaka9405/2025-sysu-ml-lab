# 《2025 Fall 机器学习与数据挖掘》 作业 1
## 实验内容
根据提供的数据，训练⼀个采⽤在不同的核函数的⽀持向量机 SVM 的 2 分类器，并验证其在测试数据集上的性能。
要求:
1) 考虑两种不同的核函数：i) 线性核函数; ii) ⾼斯核函数
2) 可以直接调⽤现成 SVM 软件包来实现
3) ⼿动实现采⽤ hinge loss 和 cross-entropy loss 的线性分类模型，并⽐较它们的优劣
## 实验原理
在线性分类器中，决策边界总是一个超平面。目标是找到能够分离不同类型样本的超平面。

定义分类损失为误分类的样本数量，在这种情况下，如果样本是线性可分的，将会有无数个超平面满足分类损失为零。

直觉上来说，假如决策边界非常贴近样本边界，认为训练是过拟合的，模型在未见过的测试集上表现未必好。因此需要在所有待选择的超平面中，选择一个既能区分两类样本，又离两类样本尽可能远的超平面。间隔越大，分类就越稳健，这种情况下，即便新增一个样本（测试集），也不容易分错。
### 硬间隔 SVM
数学上看，假设有训练集 $\left\{{(\mathbf{x}_n,y_n)}\right\}$，其中 $\mathbf{x}_i\in \mathbb{R}^d,\ y_i\in \left\{{-1,+1}\right\}$，SVM 的原理是找到一个超平面 $\mathbf{w}^T \mathbf{x}+b=0$，使得所有正类样本 （$y_i=+1$）满足 $\mathbf{w}^T \mathbf{x}+b\ge 1$，对所有负类样本（$y_i=-1$）满足 $\mathbf{w}^T \mathbf{x}+b\le -1$。称这两个边界构成的区域为间隔带，其宽度定义为 $\dfrac{2}{\Vert \mathbf{w}\Vert}$。SVM 的目标是最大化间隔带的宽度，即最小化 $\Vert \mathbf{w}\Vert$ 的值。

当样本线性可分时，最大间隔超平面可以通过求解下面的优化问题得到：
$$\begin{align}&\min_{\mathbf{w},b} \frac{1}{2}\Vert \mathbf{w}\Vert ^2 \\[4pt]
&\quad \text{s.t. }y_i(\mathbf{w}^T \mathbf{x}+b)\ge 1,\quad i=1,2,\cdots ,n
\end{align}$$
这是一个二次规划问题，考虑其对偶形式，拉格朗日函数为：
$$\mathcal{L}(\mathbf{x},b,\alpha)=\frac{1}{2}\Vert \mathbf{w}\Vert-\sum_{i=1}^{n}\alpha_i[y_i(\mathbf{w}^T \mathbf{x}_i+b)-1],\quad(\alpha_i\ge 0)$$
对 $\mathbf{w}$ 和 $b$ 分别求偏导并令其为 $0$，得到 $$\mathbf{w}=\sum_{i=1}^{n}\alpha_i y_i \mathbf{x}_i,\quad \sum_{i=1}^{n}\alpha_i y_i=0$$
代入原式得到对偶问题：
$$\begin{align}
&\max_\alpha \sum_{i=1}^{n}\alpha_i-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \\[4pt]
&\quad \text{s.t. }\alpha_i\ge 0,\ \sum_{i}^{n}\alpha_i y_i=0
\end{align}$$
这也是一个二次规划问题，相比原问题（需要优化的参数量等于特征维度 $d$），对偶问题需要优化的参数量等于样本数量 $n$，在 $d\gg n$ 的情况下求解对偶问题显然更高效。

对偶问题还具有以下性质：只有满足 $y_i(\mathbf{w}^T \mathbf{x}_i+b)=1$ 对应的 $\alpha_i$ 满足 $\alpha_i>0$，称这些满足 $\alpha_i>0$ 的样本点为支持向量 $\mathrm{SV}$。支持向量一定位于决策边界上，因此，在对一个未见过的样本进行分类时，只需要计算它与支持向量的相似度：
$$f(\mathbf{x})=\mathrm{sign}\left({\sum_{i\in \mathrm{SV}}}\alpha_i y_i \mathbf{x}_i^T \mathbf{x}+b\right)$$
可以证明支持向量的个数是稀疏的，这个性质可以大大降低计算复杂度。
### 软间隔 SVM
当样本线性不可分或存在噪声时，上述最优化问题不存在可行解，为了解决这个问题，引入松弛变量 $\xi_i\ge 0$，允许部分样本违反约束：
$$\begin{align}&\min_{\mathbf{w},b} \frac{1}{2}\Vert \mathbf{w}\Vert ^2 +C \sum_{i=1}^{n}\xi_i \\[4pt]
&\quad \text{s.t. }y_i(\mathbf{w}^T \mathbf{x}+b)\ge 1-\xi_i,\ \xi_i\ge 0
\end{align}$$
其中 $C$ 是正则化参数。其对偶形式变为：
$$\begin{align}
&\max_\alpha \sum_{i=1}^{n}\alpha_i-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \\[4pt]
&\quad \text{s.t. }0\le\alpha_i\le C,\ \sum_{i}^{n}\alpha_i y_i=0
\end{align}$$
### 非线性化与核函数
无论是硬间隔 SVM 还是软间隔 SVM 都只能处理线性问题，对于非线性问题，一种方法是通过变换将数据映射到高维特征空间，使其在高维空间中线性可分。

记这样的函数为基函数 $\phi:x\rightarrow \phi(x)$，最优化问题变为
$$\begin{align}&\min_{\mathbf{w},b} \frac{1}{2}\Vert \mathbf{w}\Vert ^2 +C \sum_{i=1}^{n}\xi_i \\[4pt]
&\quad \text{s.t. }y_i(\mathbf{w}^T \phi(\mathbf{x})+b)\ge 1-\xi_i,\ \xi_i\ge 0
\end{align}$$
决策函数变为：
$$f(\mathbf{x})=\mathrm{sign}\left({\sum_{i\in \mathrm{SV}}}\alpha_i y_i \phi(\mathbf{x}_i)^T \phi(\mathbf{x})+b\right)$$ 但在高维空间下计算 $\phi(\mathbf{x}_i)$ 与 $\phi(\mathbf{x})$ 的内积是昂贵的，因此引入核技巧：若存在一个函数 $k$ 满足 $k(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^T\phi(\mathbf{x}')$ ，就不需要计算 $\phi(\mathbf{x})$。

本次实验用到的两个核函数有：
1. 线性核：$k(\mathbf{x},\mathbf{x}')=\mathbf{x}^T \mathbf{x'}$，相当于不使用核函数。
2. 高斯核：$k(\mathbf{x},\mathbf{x}')=\exp \left\{{-\dfrac{1}{2\sigma^2}\Vert \mathbf{x}-\mathbf{x}'\Vert^2}\right\}$，也称 RBF 核，是最常用的非线性核，它对应于一个无限维的特征空间。
## 实验过程
提供的数据为完整 MNIST 数据集中提取的简化版本，仅包含数字 8 和 9。

使用`sklearn`包进行 SVM 实验，实验过程为：
1. 读取数据集
2. 将每张二维图片 $(N,28,28)$ 展平成一维特征 $(N,784)$ 
3. 对输入数据归一化，提高准确率
4. 构建两个分别使用线性核函数和高斯核函数的 SVM，分别进行训练和测试

代码如下：
```python
dataset_root = Path(r'C:\Temp\processed_8_9')
training_dataset = torch.load(dataset_root / 'training.pt')
testing_dataset = torch.load(dataset_root / 'test.pt')

X_train, y_train = training_dataset
X_test, y_test = testing_dataset

X_train = X_train.reshape(X_train.shape[0], -1).float()  # (N, 784)
X_test = X_test.reshape(X_test.shape[0], -1).float()

torch.nn.functional.normalize(X_train, p=2.0, dim=1, out=X_train)
torch.nn.functional.normalize(X_test, p=2.0, dim=1, out=X_test)

y_train = (y_train - 8).float()  # (N,)
y_test = (y_test - 8).float()

clf = SVC(C=10, kernel='linear') # linear kernel
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f'Test accuracy (SVM linear kernel): {accuracy * 100:.2f}%')

clf = SVC(C=10, kernel='rbf') # rbf kernel
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f'Test accuracy (SVM rbf kernel): {accuracy * 100:.2f}%')
```
运行结果：
```
Test accuracy (SVM linear kernel): 98.23%
Test accuracy (SVM rbf kernel): 99.75%
```
可见对于手写文本分类这种非线性问题，使用高斯核函数的准确率要显著高于线性核函数。

值得注意的是线性核函数 SVM 对正则项 $C$ 的变化相比 RBF 核要敏感，经过试验后认为 $C=10$ 是较优的超参数。

为了与 SVM 模型比较，实验还分别额外构建了基于 hinge loss 和 cross-entropy loss 的线性分类模型。

hinge loss 公式为：$\mathcal{L}=\max(0,1-y\cdot f(x))$，其中 $y\in \left\{{-1,+1}\right\}$

cross-entropy loss 公式为：$\mathcal{L}=-y\log p+(1-y)\log(1-p)$，其中 $y\in \left\{{0,1}\right\},\ p=\sigma(f(x))$ 

两个不同的损失函数决定了模型优化的方向：hinge loss 要求尽可能将两种样本区分开，$y=+1$ 时要求 $f(x)\ge 1$，$y=-1$ 时要求 $f(x)\le -1$，仅对分类错误的样本或 $f(x)\in (-1,1)$ 的不确定样本有惩罚。cross-entropy loss 则在最大似然估计上对模型进行优化。

实际上 hinge loss 在 SVM 中也得到应用，考虑上文软间隔 SVM 的优化目标：
$$\begin{align}&\min_{\mathbf{w},b} \frac{1}{2}\Vert \mathbf{w}\Vert ^2 +C \sum_{i=1}^{n}\xi_i \\[4pt]
&\quad \text{s.t. }y_i(\mathbf{w}^T \mathbf{x}+b)\ge 1-\xi_i,\ \xi_i\ge 0
\end{align}$$
将其写成无约束优化问题的形式：
$$\min_{\mathbf{w},b} \frac{1}{2}\Vert \mathbf{w}\Vert ^2 +C \sum_{i=1}^{n}\max \left\{{0,1-y_i(\mathbf{w}^T \mathbf{x}+b)}\right\}$$ 本质上就是 hinge loss + L2 正则项。

使用 pytorch 构建线性模型的实现如下：
```python
class LinearClassifierModel(torch.nn.Module):
	def __init__(self, input_dim):
		super(LinearClassifierModel, self).__init__()
		self.linear = torch.nn.Linear(input_dim, 1)
		torch.nn.init.kaiming_normal_(self.linear.weight) # 初始化

	def forward(self, x):
		return self.linear(x)


def lcm_train(
	X_train,
	y_train,
	epochs=100,
	lr=0.01,
	criterion: Literal['hinge', 'cross_entropy'] = 'hinge',
):
	if criterion == 'hinge':
		y_train = y_train * 2 - 1
	model = LinearClassifierModel(X_train.shape[1])
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

	if criterion == 'hinge':
		loss_fn = lambda y_pred, y_true: torch.mean(torch.clamp(1 - y_true * y_pred, min=0))
	elif criterion == 'cross_entropy':
		loss_fn = torch.nn.BCEWithLogitsLoss() # sigmoid + BCE
	else:
		raise ValueError('Unsupported criterion')

	model.train()
	for epoch in range(epochs):
		optimizer.zero_grad()
		outputs = model(X_train).squeeze()

		loss = loss_fn(outputs, y_train)

		loss.backward()
		optimizer.step()

	return model


def lcm_test(model, X_test, y_test, criterion: Literal['hinge', 'cross_entropy'] = 'hinge'):
	model.eval()
	with torch.no_grad():
		outputs = model(X_test).squeeze()
		if criterion == 'hinge':
			predictions = torch.where(outputs >= 0, torch.tensor(1.0), torch.tensor(0.0))
		else:
			probs = torch.sigmoid(outputs)
			predictions = torch.where(probs > 0.5, torch.tensor(1.0), torch.tensor(0.0))

		accuracy = torch.mean((predictions == y_test).float()).item()
	return accuracy
```

其中优化器使用了 AdamW，它通过维护梯度的一阶矩和二阶矩的移动平均值来动态调整每个参数的学习率。

外部测试代码为：
```python
model = lcm.lcm_train(
	X_train,
	y_train,
	epochs=100,
	lr=0.02,
	criterion='hinge',
)
accuracy = lcm.lcm_test(model, X_test, y_test, criterion='hinge')
print(f'Test accuracy (LCM hinge loss): {accuracy * 100:.2f}%')

model = lcm.lcm_train(
	X_train,
	y_train,
	epochs=200,
	lr=0.01,
	criterion='cross_entropy',
)
accuracy = lcm.lcm_test(model, X_test, y_test, criterion='cross_entropy')
print(f'Test accuracy (LCM cross entropy): {accuracy * 100:.2f}%')

```

运行结果：
```
Test accuracy (LCM hinge loss): 97.78%
Test accuracy (LCM cross entropy): 97.53%
```
准确率均不及使用 SVM 的两个实验组，在预期之中。使用 hinge loss 的模型准确率稍高，个人认为在误差范围内。

添加 L2 正则化后（理论上此时使用 hinge loss 的模型在优化目标上和使用线性核的软 SVM 等价）的分类准确率有所降低，可能参数没调好，干脆不要正则化了。

## 实验结果
经过四组实验（线性核 SVM、高斯核 SVM、hinge loss 线性模型、cross-entropy loss 线性模型）的比较，认为使用高斯核的 SVM 在分类准确率上具有显著优势，线性核 SVM 与 hinge loss 线性模型其次，其中线性核 SVM 可能在最优化的数值求解算法上具有优势，因此准确率在二者比较中占优，使用 cross-entropy loss 的线性模型分类准确率最差。本次实验证明了 SVM 在优化非线性问题时相比传统线性模型具有显著优势，为后续的深度学习实验打下坚实基础。