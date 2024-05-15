#! https://zhuanlan.zhihu.com/p/697889538

# 机器学习中的概率（1）——多元高斯分布

这个系列写给对机器学习感兴趣，但对概率论不熟悉的同学。

本文的主要内容如下：

- 前置知识
  - 随机向量
  - 随机向量的线性变换
- 高斯随机变量和多元正态分布的定义
  - 三个等价定义
  - 联合密度函数
- 联合分布服从多元正态分布时的性质
  - 每个随机变量都服从正态分布
  - 不相关时相互独立

本文不会涉及艰深的证明，也因此不会特别严谨。阅读本文只需读者了解概率论的基础知识，如协方差、概率密度函数、正态分布。让我们开始吧。

# 前置知识

## 随机变量与随机向量

在讨论多元正态分布之前，我们先来看看一些前置知识，它们在定义多元正态分布时起了重要作用。

我们都学过向量的相关知识，它允许我们紧凑地表示多个实数。但为什么一定要把元素类型限制为实数呢？
如果我们允许随机变量作为向量的元素，那我们就可以得到随机向量。

随机向量是一个包含多个随机变量的列表，我们可以把它视为一个  <img src="https://www.zhihu.com/equation?tex= n \times 1 " alt=" n \times 1 " class="ee_img tr_noresize" eeimg="1">  的列向量。


<img src="https://www.zhihu.com/equation?tex=\textbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_n \end{bmatrix}
" alt="\textbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_n \end{bmatrix}
" class="ee_img tr_noresize" eeimg="1">

为了方便表示，我们有时也会把随机向量写为  <img src="https://www.zhihu.com/equation?tex=\textbf{X} = (X_1, X_2, \cdots, X_n)^T" alt="\textbf{X} = (X_1, X_2, \cdots, X_n)^T" class="ee_img tr_noresize" eeimg="1"> 。

与随机变量有均值、方差类似，随机向量也有均值向量和协方差矩阵。（想一想，怎么定义它们？）

 <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的均值向量为  <img src="https://www.zhihu.com/equation?tex=\boldsymbol\mu = (\mu_1, \mu_2, \cdots, \mu_n)^T" alt="\boldsymbol\mu = (\mu_1, \mu_2, \cdots, \mu_n)^T" class="ee_img tr_noresize" eeimg="1"> ，其中  <img src="https://www.zhihu.com/equation?tex=\mu_i" alt="\mu_i" class="ee_img tr_noresize" eeimg="1">  是随机变量  <img src="https://www.zhihu.com/equation?tex=X_i" alt="X_i" class="ee_img tr_noresize" eeimg="1">  的均值。

 <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的协方差矩阵为

<img src="https://www.zhihu.com/equation?tex=\Sigma = \begin{bmatrix} Cov(X_1, X_1) & Cov(X_1, X_2) & \cdots & Cov(X_1, X_n) \\ Cov(X_2, X_1) & Cov(X_2, X_2) & \cdots & Cov(X_2, X_n) \\ \vdots & \vdots & \ddots & \vdots \\ Cov(X_n, X_1) & Cov(X_n, X_2) & \cdots & Cov(X_n, X_n) \end{bmatrix}" alt="\Sigma = \begin{bmatrix} Cov(X_1, X_1) & Cov(X_1, X_2) & \cdots & Cov(X_1, X_n) \\ Cov(X_2, X_1) & Cov(X_2, X_2) & \cdots & Cov(X_2, X_n) \\ \vdots & \vdots & \ddots & \vdots \\ Cov(X_n, X_1) & Cov(X_n, X_2) & \cdots & Cov(X_n, X_n) \end{bmatrix}" class="ee_img tr_noresize" eeimg="1">

其中  <img src="https://www.zhihu.com/equation?tex=Cov(X_i, X_j)" alt="Cov(X_i, X_j)" class="ee_img tr_noresize" eeimg="1">  是随机变量  <img src="https://www.zhihu.com/equation?tex=X_i" alt="X_i" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=X_j" alt="X_j" class="ee_img tr_noresize" eeimg="1">  的协方差。特别地，当  <img src="https://www.zhihu.com/equation?tex=i=j" alt="i=j" class="ee_img tr_noresize" eeimg="1">  时， <img src="https://www.zhihu.com/equation?tex=Cov(X_i, X_j) = Var(X_i)" alt="Cov(X_i, X_j) = Var(X_i)" class="ee_img tr_noresize" eeimg="1">  是随机变量  <img src="https://www.zhihu.com/equation?tex=X_i" alt="X_i" class="ee_img tr_noresize" eeimg="1">  的方差。

不难发现，协方差矩阵是一个对称矩阵，证明给读者留作练习。

下面是一个小练习，读者可以测试一下自己对概念的理解：

设随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{X} = (X_1, X_2)^T" alt="\textbf{X} = (X_1, X_2)^T" class="ee_img tr_noresize" eeimg="1"> ，其均值向量为  <img src="https://www.zhihu.com/equation?tex=\mu = (0, 0)^T" alt="\mu = (0, 0)^T" class="ee_img tr_noresize" eeimg="1"> ，协方差矩阵为  <img src="https://www.zhihu.com/equation?tex=\Sigma = \begin{bmatrix}  1 & 2.4 \\ x & 9 \end{bmatrix}" alt="\Sigma = \begin{bmatrix}  1 & 2.4 \\ x & 9 \end{bmatrix}" class="ee_img tr_noresize" eeimg="1"> 。

1. 计算  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  的值。
2. 计算 <img src="https://www.zhihu.com/equation?tex=Corr(X_1, X_2)" alt="Corr(X_1, X_2)" class="ee_img tr_noresize" eeimg="1"> 。

注： <img src="https://www.zhihu.com/equation?tex=Corr(X_1, X_2) = \frac{Cov(X_1, X_2)} {\sqrt{Var(X_1) \times Var(X_2)}}" alt="Corr(X_1, X_2) = \frac{Cov(X_1, X_2)} {\sqrt{Var(X_1) \times Var(X_2)}}" class="ee_img tr_noresize" eeimg="1"> 
，表示  <img src="https://www.zhihu.com/equation?tex=X_1" alt="X_1" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=X_2" alt="X_2" class="ee_img tr_noresize" eeimg="1">  之间的相关系数。

## 随机向量的线性变换

类似实向量通过线性变换可以得到新的实向量，随机向量也可以进行线性变换得到新的随机向量。

设  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=m \times n" alt="m \times n" class="ee_img tr_noresize" eeimg="1">  的实矩阵， <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  维随机向量， <img src="https://www.zhihu.com/equation?tex=\bold{b}" alt="\bold{b}" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1">  维实向量，那么  <img src="https://www.zhihu.com/equation?tex=A\textbf{X} + \bold{b}" alt="A\textbf{X} + \bold{b}" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1">  维随机向量。

如果设

<img src="https://www.zhihu.com/equation?tex=\textbf{Y} = A\textbf{X} + \bold{b}" alt="\textbf{Y} = A\textbf{X} + \bold{b}" class="ee_img tr_noresize" eeimg="1">

那么可以发现


<img src="https://www.zhihu.com/equation?tex=\begin{align*}
\textbf{Y}
&= \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n}  \\ a_{21} & a_{22} & \cdots & a_{2n} &  \\ \vdots & \vdots & \ddots & \vdots &  \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}
\begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_n \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}
\\
&= \begin{bmatrix} A_{1*} \\ A_{2*} \\ \vdots \\ A_{m*} \end{bmatrix} \textbf{X} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}
（分块矩阵）
\\
&= \begin{bmatrix} A_{1*} X + b_1 \\ A_{2*} X + b_2 \\ \vdots \\ A_{m*} X + b_m \end{bmatrix}
\\
&= A\textbf{X} + \bold{b}
\end{align*}
" alt="\begin{align*}
\textbf{Y}
&= \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n}  \\ a_{21} & a_{22} & \cdots & a_{2n} &  \\ \vdots & \vdots & \ddots & \vdots &  \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}
\begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_n \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}
\\
&= \begin{bmatrix} A_{1*} \\ A_{2*} \\ \vdots \\ A_{m*} \end{bmatrix} \textbf{X} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}
（分块矩阵）
\\
&= \begin{bmatrix} A_{1*} X + b_1 \\ A_{2*} X + b_2 \\ \vdots \\ A_{m*} X + b_m \end{bmatrix}
\\
&= A\textbf{X} + \bold{b}
\end{align*}
" class="ee_img tr_noresize" eeimg="1">

其中  <img src="https://www.zhihu.com/equation?tex=A_{i*}" alt="A_{i*}" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  的第  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  行的行向量。进而有


<img src="https://www.zhihu.com/equation?tex=\begin{align*}
Y_i
&= A_{i*} \textbf{X} + b_i \\
&= \sum_{k=1}^n a_{ik} X_k + b_i \\
\end{align*}
" alt="\begin{align*}
Y_i
&= A_{i*} \textbf{X} + b_i \\
&= \sum_{k=1}^n a_{ik} X_k + b_i \\
\end{align*}
" class="ee_img tr_noresize" eeimg="1">

从上式可以发现，如果我们取新的随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{Y}" alt="\textbf{Y}" class="ee_img tr_noresize" eeimg="1">  的一个分量  <img src="https://www.zhihu.com/equation?tex=Y_i" alt="Y_i" class="ee_img tr_noresize" eeimg="1"> ，那么它可以被表示为原来的随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的分量的线性组合加上一个常数。

下面让我们尝试找找  <img src="https://www.zhihu.com/equation?tex=\textbf{Y}" alt="\textbf{Y}" class="ee_img tr_noresize" eeimg="1">  的均值向量  <img src="https://www.zhihu.com/equation?tex=\boldsymbol\mu_Y" alt="\boldsymbol\mu_Y" class="ee_img tr_noresize" eeimg="1">  和协方差矩阵  <img src="https://www.zhihu.com/equation?tex=\Sigma_Y" alt="\Sigma_Y" class="ee_img tr_noresize" eeimg="1">  与  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的关系。

设  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的均值向量为  <img src="https://www.zhihu.com/equation?tex=\boldsymbol\mu_X" alt="\boldsymbol\mu_X" class="ee_img tr_noresize" eeimg="1"> ，协方差矩阵为  <img src="https://www.zhihu.com/equation?tex=\Sigma_X" alt="\Sigma_X" class="ee_img tr_noresize" eeimg="1"> ，则  <img src="https://www.zhihu.com/equation?tex=A\textbf{X}" alt="A\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的均值向量为  <img src="https://www.zhihu.com/equation?tex=A\boldsymbol\mu_X + \bold b" alt="A\boldsymbol\mu_X + \bold b" class="ee_img tr_noresize" eeimg="1"> ，协方差矩阵为  <img src="https://www.zhihu.com/equation?tex=A\Sigma_X A^T" alt="A\Sigma_X A^T" class="ee_img tr_noresize" eeimg="1"> 。

对均值向量的关系的证明给读者留作练习，我们这里只证明协方差矩阵的关系：


<img src="https://www.zhihu.com/equation?tex=\Sigma_Y = A\Sigma_X A^T
" alt="\Sigma_Y = A\Sigma_X A^T
" class="ee_img tr_noresize" eeimg="1">

证明：


<img src="https://www.zhihu.com/equation?tex=\begin{align*}
Cov(Y_i, Y_j) &= Cov(A_{i*} \textbf{X} + b_i, A_{j*} \textbf{X} + b_j) \\
&= Cov(\sum_{k=1}^n a_{ik} X_k, \sum_{l=1}^n a_{jl} X_l) （增减常数协方差不变）\\
&= \sum_{k=1}^n\sum_{l=1}^n a_{ik} a_{jl} Cov(X_k, X_l) \\
&= A_{i*} \Sigma_{X} A^T_{*j} \\

\end{align*}
" alt="\begin{align*}
Cov(Y_i, Y_j) &= Cov(A_{i*} \textbf{X} + b_i, A_{j*} \textbf{X} + b_j) \\
&= Cov(\sum_{k=1}^n a_{ik} X_k, \sum_{l=1}^n a_{jl} X_l) （增减常数协方差不变）\\
&= \sum_{k=1}^n\sum_{l=1}^n a_{ik} a_{jl} Cov(X_k, X_l) \\
&= A_{i*} \Sigma_{X} A^T_{*j} \\

\end{align*}
" class="ee_img tr_noresize" eeimg="1">

至此，读者应当对随机向量已经有了基本的了解，知道了随机向量的均值向量、协方差矩阵，以及其线性变换的定义。接下来，我们只讨论连续型随机变量和连续型随机向量。

# 定义

终于，我们进入了这篇文章的主题——高斯随机向量和多元正态分布。读者应该都对正态分布很熟悉，或许也看过二维多元正态分布的联合密度函数的图像，这是一个很好的起点。如果读者没有看过图像，那现在你也看过了：

![Image](https://pic4.zhimg.com/80/v2-7ea0ad25d45551c7e9b590833a636ca1.png)

## 高斯随机向量的定义

我们先来定义高斯随机向量。

令  <img src="https://www.zhihu.com/equation?tex=Z_1, Z_2, \cdots, Z_n" alt="Z_1, Z_2, \cdots, Z_n" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  个相互独立的服从正态分布的随机变量，即  <img src="https://www.zhihu.com/equation?tex=Z_i \sim N(0, 1)" alt="Z_i \sim N(0, 1)" class="ee_img tr_noresize" eeimg="1">  ，则它们可以构成一个  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  维的随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{Z}" alt="\textbf{Z}" class="ee_img tr_noresize" eeimg="1"> ，满足：


<img src="https://www.zhihu.com/equation?tex=\textbf{Z} = (Z_1, Z_2, \cdots, Z_n)^T
" alt="\textbf{Z} = (Z_1, Z_2, \cdots, Z_n)^T
" class="ee_img tr_noresize" eeimg="1">

设  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=m \times n " alt="m \times n " class="ee_img tr_noresize" eeimg="1">  的矩阵， <img src="https://www.zhihu.com/equation?tex=\bold b" alt="\bold b" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1">  维实向量，那么我们定义  <img src="https://www.zhihu.com/equation?tex=\textbf{X} = A\textbf{Z} + \bold b" alt="\textbf{X} = A\textbf{Z} + \bold b" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1">  维高斯随机向量。

在本文的剩余部分，我们将一直用  <img src="https://www.zhihu.com/equation?tex=\textbf{Z}" alt="\textbf{Z}" class="ee_img tr_noresize" eeimg="1">  来表示满足上述条件的随机向量，这里的  <img src="https://www.zhihu.com/equation?tex=Z" alt="Z" class="ee_img tr_noresize" eeimg="1">  服从的联合分布也被称为标准多元正态分布。

敏锐的读者应该会发现，这就是我们之前提到过的“随机向量的线性变换”。还记得线性变换后随机向量的均值向量和协方差矩阵如何变化吗？

请读者自证：


<img src="https://www.zhihu.com/equation?tex=\begin{align*}
\boldsymbol\mu_X &= \bold b \\
\Sigma_X &= A I A^T \\
&= A A^T
\end{align*}
" alt="\begin{align*}
\boldsymbol\mu_X &= \bold b \\
\Sigma_X &= A I A^T \\
&= A A^T
\end{align*}
" class="ee_img tr_noresize" eeimg="1">

其中  <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  维单位矩阵。

现在，让我们来直观地看看这个线性变换究竟产生了怎样的随机向量。

令  <img src="https://www.zhihu.com/equation?tex=\bold b = \begin{bmatrix} 1 \\ 1 \end{bmatrix}" alt="\bold b = \begin{bmatrix} 1 \\ 1 \end{bmatrix}" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex= A_1 = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}" alt=" A_1 = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex= A_2 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}" alt=" A_2 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}" class="ee_img tr_noresize" eeimg="1">  ，

再令  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1 = A_1\textbf{Z} + \bold b_1" alt="\textbf{X}_1 = A_1\textbf{Z} + \bold b_1" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2 = A_2\textbf{Z} + \bold b_2" alt="\textbf{X}_2 = A_2\textbf{Z} + \bold b_2" class="ee_img tr_noresize" eeimg="1">  。

那么由之前的定义，  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1" alt="\textbf{X}_1" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2" alt="\textbf{X}_2" class="ee_img tr_noresize" eeimg="1">  都是高斯随机向量。

我们对  <img src="https://www.zhihu.com/equation?tex=\textbf{Z}" alt="\textbf{Z}" class="ee_img tr_noresize" eeimg="1">  、  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1" alt="\textbf{X}_1" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2" alt="\textbf{X}_2" class="ee_img tr_noresize" eeimg="1">  分别采样 1000 次，并把它们画成散点图，得到下面三张图片：

![Image](https://pic4.zhimg.com/80/v2-266b346ca0d4400b2262177be8f5cb88.png)

可以看到， <img src="https://www.zhihu.com/equation?tex=\textbf{Z}" alt="\textbf{Z}" class="ee_img tr_noresize" eeimg="1">  的散点图接近一个圆， <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1" alt="\textbf{X}_1" class="ee_img tr_noresize" eeimg="1">  的散点图接近一个椭圆，而  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2" alt="\textbf{X}_2" class="ee_img tr_noresize" eeimg="1">  的散点图则接近一条直线。

如果希望进一步解释这种散点图呈以上形状的原因，我们就需要研究  <img src="https://www.zhihu.com/equation?tex=\textbf{Z}" alt="\textbf{Z}" class="ee_img tr_noresize" eeimg="1">  、  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1" alt="\textbf{X}_1" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2" alt="\textbf{X}_2" class="ee_img tr_noresize" eeimg="1">  各自的联合密度函数，从而分析其等高线的方程了。

## 联合密度函数

我们这里直接给出结论，并给出一个并不严谨的解释，因为严格证明超出了这篇文章的范围。

设  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  由  <img src="https://www.zhihu.com/equation?tex=\textbf{X} = A\textbf{Z} + \boldsymbol\mu" alt="\textbf{X} = A\textbf{Z} + \boldsymbol\mu" class="ee_img tr_noresize" eeimg="1">  定义的高斯随机向量，且  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  是可逆矩阵。

那么， <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的联合密度函数为：


<img src="https://www.zhihu.com/equation?tex=f_{\textbf{X}}(x_1, x_2, \cdots, x_n) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\bold x - \boldsymbol\mu)^T\Sigma^{-1}(\bold x - \boldsymbol\mu)\right)
" alt="f_{\textbf{X}}(x_1, x_2, \cdots, x_n) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\bold x - \boldsymbol\mu)^T\Sigma^{-1}(\bold x - \boldsymbol\mu)\right)
" class="ee_img tr_noresize" eeimg="1">

其中  <img src="https://www.zhihu.com/equation?tex=\bold x = (x_1, x_2, \cdots, x_n)^T" alt="\bold x = (x_1, x_2, \cdots, x_n)^T" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的一个可能的取值， <img src="https://www.zhihu.com/equation?tex=\boldsymbol\mu" alt="\boldsymbol\mu" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的均值向量， <img src="https://www.zhihu.com/equation?tex=\Sigma=AA^T" alt="\Sigma=AA^T" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的协方差矩阵。

注意这里对  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  的可逆性的要求，这保证了高斯随机向量  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  不会像之前的  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2" alt="\textbf{X}_2" class="ee_img tr_noresize" eeimg="1">  那样退化到低维。

读者可以自己尝试证明  <img src="https://www.zhihu.com/equation?tex=\textbf{X} = \textbf{Z}" alt="\textbf{X} = \textbf{Z}" class="ee_img tr_noresize" eeimg="1">  的情况。

接下来，对一般的  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  ，我们尝试（不严谨地）解释这个联合密度函数的由来。

设  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  是均值向量为  <img src="https://www.zhihu.com/equation?tex=\boldsymbol\mu" alt="\boldsymbol\mu" class="ee_img tr_noresize" eeimg="1"> ，协方差矩阵为  <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1">  的高斯随机向量，它由  <img src="https://www.zhihu.com/equation?tex=\textbf{X} = A\textbf{Z} + \boldsymbol\mu" alt="\textbf{X} = A\textbf{Z} + \boldsymbol\mu" class="ee_img tr_noresize" eeimg="1">  定义，且  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  是可逆矩阵。

因此，对随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{Z}" alt="\textbf{Z}" class="ee_img tr_noresize" eeimg="1">  的每一个可能的取值  <img src="https://www.zhihu.com/equation?tex=\bold z" alt="\bold z" class="ee_img tr_noresize" eeimg="1">  ，都对应着一个  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的一个可能的取值  <img src="https://www.zhihu.com/equation?tex=A_1\bold z + \boldsymbol\mu" alt="A_1\bold z + \boldsymbol\mu" class="ee_img tr_noresize" eeimg="1">  。同时由于  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  是可逆的，对随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的每一个可能的取值  <img src="https://www.zhihu.com/equation?tex=\bold x" alt="\bold x" class="ee_img tr_noresize" eeimg="1">  ，也对应着一个  <img src="https://www.zhihu.com/equation?tex=\textbf{Z}" alt="\textbf{Z}" class="ee_img tr_noresize" eeimg="1">  的一个可能的取值  <img src="https://www.zhihu.com/equation?tex=\bold z = A^{-1}(\bold x - \boldsymbol\mu)" alt="\bold z = A^{-1}(\bold x - \boldsymbol\mu)" class="ee_img tr_noresize" eeimg="1"> 。

那我们就可以认为：


<img src="https://www.zhihu.com/equation?tex=P(\textbf{X} \approx \bold x) \approx P(\textbf{Z} \approx \bold z)
" alt="P(\textbf{X} \approx \bold x) \approx P(\textbf{Z} \approx \bold z)
" class="ee_img tr_noresize" eeimg="1">

而又由于：


<img src="https://www.zhihu.com/equation?tex=\begin{align*}
f_{\textbf{X}}(\bold x)dx
&\approx P(\textbf{X} \approx \bold x) \\
&\approx P(\textbf{Z} \approx \bold z) \\
&\approx f_{\textbf{Z}}(\bold z)dz \\
&\approx f_{\textbf{Z}}(A^{-1}(\bold x - \boldsymbol\mu)) \frac{\partial{z}}{\partial{x}}dx \\
&\approx f_{\textbf{Z}}(A^{-1}(\bold x - \boldsymbol\mu))|det(A^{-1})|dx \\
&= \frac{f_{\textbf{Z}}(A^{-1}(\bold x - \boldsymbol\mu))}{|\Sigma|^{1/2}}dx
\end{align*}
" alt="\begin{align*}
f_{\textbf{X}}(\bold x)dx
&\approx P(\textbf{X} \approx \bold x) \\
&\approx P(\textbf{Z} \approx \bold z) \\
&\approx f_{\textbf{Z}}(\bold z)dz \\
&\approx f_{\textbf{Z}}(A^{-1}(\bold x - \boldsymbol\mu)) \frac{\partial{z}}{\partial{x}}dx \\
&\approx f_{\textbf{Z}}(A^{-1}(\bold x - \boldsymbol\mu))|det(A^{-1})|dx \\
&= \frac{f_{\textbf{Z}}(A^{-1}(\bold x - \boldsymbol\mu))}{|\Sigma|^{1/2}}dx
\end{align*}
" class="ee_img tr_noresize" eeimg="1">

因此我们有理由觉得：


<img src="https://www.zhihu.com/equation?tex=\begin{align*}
f_{\textbf{X}}(\bold x) &= \frac{f_{\textbf{Z}}(A^{-1}(\bold x - \boldsymbol\mu))}{|\Sigma|^{1/2}} \\
&= \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\bold x - \boldsymbol\mu)^T\Sigma^{-1}(\bold x - \boldsymbol\mu)\right)
\end{align*}
" alt="\begin{align*}
f_{\textbf{X}}(\bold x) &= \frac{f_{\textbf{Z}}(A^{-1}(\bold x - \boldsymbol\mu))}{|\Sigma|^{1/2}} \\
&= \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\bold x - \boldsymbol\mu)^T\Sigma^{-1}(\bold x - \boldsymbol\mu)\right)
\end{align*}
" class="ee_img tr_noresize" eeimg="1">

从而我们就得到了  <img src="https://www.zhihu.com/equation?tex=f_{\textbf{X}}(x)" alt="f_{\textbf{X}}(x)" class="ee_img tr_noresize" eeimg="1">  的表达式。

读者或许会问， <img src="https://www.zhihu.com/equation?tex=f_{\textbf{Z}}(\bold z)" alt="f_{\textbf{Z}}(\bold z)" class="ee_img tr_noresize" eeimg="1">  的表达式是怎么得到的？把文章往上翻翻，就会发现我们曾把这个结果给读者留作练习。如果读者跳过了那个练习，那现在正是补做的好时机，只要注意到  <img src="https://www.zhihu.com/equation?tex=Z_i" alt="Z_i" class="ee_img tr_noresize" eeimg="1">  相互独立，且都服从标准正态分布即可。

## 等高线

由于我们已经得到了高斯随机向量的联合密度函数，我们就可以求出其等高线了。

观察联合密度函数的表达式可知，等高线的方程总是形如：


<img src="https://www.zhihu.com/equation?tex=(\bold x - \bold \mu)^T\Sigma^{-1}(\bold x - \bold \mu) = c
" alt="(\bold x - \bold \mu)^T\Sigma^{-1}(\bold x - \bold \mu) = c
" class="ee_img tr_noresize" eeimg="1">

其中  <img src="https://www.zhihu.com/equation?tex=c" alt="c" class="ee_img tr_noresize" eeimg="1">  是常数。

结合  <img src="https://www.zhihu.com/equation?tex=\Sigma=AA^T" alt="\Sigma=AA^T" class="ee_img tr_noresize" eeimg="1">  的对称性、正定性，熟悉线性代数的读者应当能发现，这是一个高维椭球的方程。

回顾一下我们之前对  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1" alt="\textbf{X}_1" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2" alt="\textbf{X}_2" class="ee_img tr_noresize" eeimg="1">  的定义：

令  <img src="https://www.zhihu.com/equation?tex=\bold b = \begin{bmatrix} 1 \\ 1 \end{bmatrix}" alt="\bold b = \begin{bmatrix} 1 \\ 1 \end{bmatrix}" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex= A_1 = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}" alt=" A_1 = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex= A_2 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}" alt=" A_2 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}" class="ee_img tr_noresize" eeimg="1">  ，

再令  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1 = A_1\textbf{Z} + \bold b_1" alt="\textbf{X}_1 = A_1\textbf{Z} + \bold b_1" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2 = A_2\textbf{Z} + \bold b_2" alt="\textbf{X}_2 = A_2\textbf{Z} + \bold b_2" class="ee_img tr_noresize" eeimg="1">  。

这就解释了之前为什么  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1" alt="\textbf{X}_1" class="ee_img tr_noresize" eeimg="1">  的散点图是椭圆。而简单地写出  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2" alt="\textbf{X}_2" class="ee_img tr_noresize" eeimg="1">  的联合分布就能解释其散点图为什么是一条直线。

现在，让我们看看  <img src="https://www.zhihu.com/equation?tex=\textbf{Z}" alt="\textbf{Z}" class="ee_img tr_noresize" eeimg="1">  、  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1" alt="\textbf{X}_1" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2" alt="\textbf{X}_2" class="ee_img tr_noresize" eeimg="1">  各自的联合密度函数在三维空间中的形状吧：

![Image](https://pic4.zhimg.com/80/v2-21117d645818d871815daa3308f84046.png)

## 三个等价定义

接下来，读者会注意到这里的定义 1 与我们之前给出的定义略有差别，这体现在我们要求  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  是可逆矩阵上。这是因为这里给出的是多元正态分布的定义，而之前给出的是高斯随机向量的定义。它们几乎完全相同，高斯随机变量只是包含了退化到低维的情况。

也希望读者在看到多元正态分布和高斯随机向量的性质时，想想它们的性质是否互通。

**定义 1**
如果随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  由  <img src="https://www.zhihu.com/equation?tex=\textbf{X} = A\textbf{Z} + \boldsymbol\mu" alt="\textbf{X} = A\textbf{Z} + \boldsymbol\mu" class="ee_img tr_noresize" eeimg="1">  定义，
其中  <img src="https://www.zhihu.com/equation?tex=\textbf{Z}=(Z_1, Z_2, \cdots, Z_n)^T" alt="\textbf{Z}=(Z_1, Z_2, \cdots, Z_n)^T" class="ee_img tr_noresize" eeimg="1">  ，  <img src="https://www.zhihu.com/equation?tex=Z_i" alt="Z_i" class="ee_img tr_noresize" eeimg="1">  是独立同分布的标准正态随机变量， <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=n \times n" alt="n \times n" class="ee_img tr_noresize" eeimg="1">  的可逆矩阵， <img src="https://www.zhihu.com/equation?tex=\boldsymbol\mu" alt="\boldsymbol\mu" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  维实向量，那么  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  服从多元正态分布。

**定义 2**
如果随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的联合密度函数为：


<img src="https://www.zhihu.com/equation?tex=f_{\textbf{X}}(x_1, x_2, \cdots, x_n) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\bold x - \boldsymbol\mu)^T\Sigma^{-1}(\bold x - \boldsymbol\mu)\right)
" alt="f_{\textbf{X}}(x_1, x_2, \cdots, x_n) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\bold x - \boldsymbol\mu)^T\Sigma^{-1}(\bold x - \boldsymbol\mu)\right)
" class="ee_img tr_noresize" eeimg="1">

其中  <img src="https://www.zhihu.com/equation?tex=\bold x = (x_1, x_2, \cdots, x_n)^T" alt="\bold x = (x_1, x_2, \cdots, x_n)^T" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的一个可能的取值， <img src="https://www.zhihu.com/equation?tex=\boldsymbol\mu" alt="\boldsymbol\mu" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的均值向量， <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的协方差矩阵，那么  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  服从多元正态分布。

**定义 3**
如果  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的分量的所有线性组合都是正态分布的，即


<img src="https://www.zhihu.com/equation?tex=\sum_{i=1}^n a_i\textbf{Z}_i \sim N(\boldsymbol\mu, \Sigma)
" alt="\sum_{i=1}^n a_i\textbf{Z}_i \sim N(\boldsymbol\mu, \Sigma)
" class="ee_img tr_noresize" eeimg="1">

那么  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  服从多元正态分布。

我们不会给出这三个定义的等价性的证明，因为这远远超出了这篇文章的范围。

作为练习，请读者证明定义 1 可以推出定义 3，并判断我们之前给出的  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_1" alt="\textbf{X}_1" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_2" alt="\textbf{X}_2" class="ee_img tr_noresize" eeimg="1">  是否服从多元正态分布。

# 服从多元正态分布的随机变量的分量

最后，我们来讨论一下服从多元正态分布的随机变量的分量的一些性质。

## 分量是标准正态分布随机变量的线性组合

由定义 1， <img src="https://www.zhihu.com/equation?tex=\textbf{X}=A\textbf{Z}+\boldsymbol\mu" alt="\textbf{X}=A\textbf{Z}+\boldsymbol\mu" class="ee_img tr_noresize" eeimg="1">  ，从而 <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的分量  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_i" alt="\textbf{X}_i" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=\textbf{Z}" alt="\textbf{Z}" class="ee_img tr_noresize" eeimg="1">  的分量的一个线性组合加上一个常数：


<img src="https://www.zhihu.com/equation?tex=\begin{align*}
X_i
&= A_{i*} Z + \mu_i \\
&= \sum_{k=1}^n a_{ik} Z_k + \mu_i \\
\end{align*}
" alt="\begin{align*}
X_i
&= A_{i*} Z + \mu_i \\
&= \sum_{k=1}^n a_{ik} Z_k + \mu_i \\
\end{align*}
" class="ee_img tr_noresize" eeimg="1">

从而， <img src="https://www.zhihu.com/equation?tex=\textbf{X}_i" alt="\textbf{X}_i" class="ee_img tr_noresize" eeimg="1">  也服从正态分布。

## 分量正态不代表向量正态

那么一个自然的问题就是，如果随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  的分量  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_i" alt="\textbf{X}_i" class="ee_img tr_noresize" eeimg="1">  都服从正态分布，那么  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  一定服从多元正态分布吗？

答案是否定的。

举个例子，设


<img src="https://www.zhihu.com/equation?tex=\begin{align*}
\textbf{Y}_1 &\sim N(0, 1)
\\
\textbf{U} &=
\begin{cases}
1 & \text{with probability } \frac{1}{2}, \\
-1 & \text{with probability } \frac{1}{2}.
\end{cases}
\\
\textbf{Y}_2 &= \textbf{U} \cdot \textbf{X}_1
\end{align*}
" alt="\begin{align*}
\textbf{Y}_1 &\sim N(0, 1)
\\
\textbf{U} &=
\begin{cases}
1 & \text{with probability } \frac{1}{2}, \\
-1 & \text{with probability } \frac{1}{2}.
\end{cases}
\\
\textbf{Y}_2 &= \textbf{U} \cdot \textbf{X}_1
\end{align*}
" class="ee_img tr_noresize" eeimg="1">

显然  <img src="https://www.zhihu.com/equation?tex=\textbf{Y}_1" alt="\textbf{Y}_1" class="ee_img tr_noresize" eeimg="1">  服从正态分布，读者可以自己验证  <img src="https://www.zhihu.com/equation?tex=\textbf{Y}_2" alt="\textbf{Y}_2" class="ee_img tr_noresize" eeimg="1">  服从正态分布。

而它们构成的随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{Y} = (\textbf{Y}_1, \textbf{Y}_2)^T" alt="\textbf{Y} = (\textbf{Y}_1, \textbf{Y}_2)^T" class="ee_img tr_noresize" eeimg="1">  服从多元正态分布吗？

我们知道，如果一个随机向量服从多元正态分布，那么它的分量的线性组合服从正态分布（为什么？），而  <img src="https://www.zhihu.com/equation?tex=\textbf{Y}_1 + \textbf{Y}_2" alt="\textbf{Y}_1 + \textbf{Y}_2" class="ee_img tr_noresize" eeimg="1">  却不服从正态分布。（请读者自行验证）因此， <img src="https://www.zhihu.com/equation?tex=\textbf{Y}" alt="\textbf{Y}" class="ee_img tr_noresize" eeimg="1">  不服从多元正态分布。

![Image](https://pic4.zhimg.com/80/v2-dbd210c552c23e19b225ab4aeab3fa42.png)

下面给出  <img src="https://www.zhihu.com/equation?tex=\textbf{Y}" alt="\textbf{Y}" class="ee_img tr_noresize" eeimg="1">  的联合密度函数的图像：

![Image](https://pic4.zhimg.com/80/v2-53429231cd411e48270f50e4fd1001ab.png)

这个例子提示我们，即使一个随机向量的每个分量都服从正态分布，它也不一定服从多元正态分布。

## 正态随机变量不相关时相互独立

我们再给出一个重要的性质：

如果随机向量  <img src="https://www.zhihu.com/equation?tex=\textbf{X}" alt="\textbf{X}" class="ee_img tr_noresize" eeimg="1">  服从多元正态分布，且对任意的  <img src="https://www.zhihu.com/equation?tex=i, j" alt="i, j" class="ee_img tr_noresize" eeimg="1"> ，有  <img src="https://www.zhihu.com/equation?tex=Cov(X_i, X_j) = 0" alt="Cov(X_i, X_j) = 0" class="ee_img tr_noresize" eeimg="1">  ，那么这  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  个分量对应的随机变量相互独立。

这个性质并不平凡。我们知道如果两个随机变量相互独立，那其相关系数为 0 ，但反过来不一定成立。因为相关系数仅在反映了随机变量的线性相关性，而随机变量可以有除了线性相关以外的相关性。

这个性质的证明并不困难，只要注意到当  <img src="https://www.zhihu.com/equation?tex=Cov(X_i, X_j) = 0" alt="Cov(X_i, X_j) = 0" class="ee_img tr_noresize" eeimg="1">  ，协方差矩阵  <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1">  变为一个对角矩阵，从而有：


<img src="https://www.zhihu.com/equation?tex=\begin{align*}
f_{\textbf{X}}(x_1, x_2, \cdots, x_n)
&= \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\bold x - \boldsymbol\mu)^T\Sigma^{-1}(\bold x - \boldsymbol\mu)\right) \\
&= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right) \\
&= \prod_{i=1}^n f_{\textbf{X}_i}(x_i)
\end{align*}
" alt="\begin{align*}
f_{\textbf{X}}(x_1, x_2, \cdots, x_n)
&= \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\bold x - \boldsymbol\mu)^T\Sigma^{-1}(\bold x - \boldsymbol\mu)\right) \\
&= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right) \\
&= \prod_{i=1}^n f_{\textbf{X}_i}(x_i)
\end{align*}
" class="ee_img tr_noresize" eeimg="1">

因此，对任意一组联合分布服从多元正态分布的随机变量  <img src="https://www.zhihu.com/equation?tex=\textbf{X}_i" alt="\textbf{X}_i" class="ee_img tr_noresize" eeimg="1">  ，如果其两两的协方差为 0，那么它们相互独立。

文末，我们给读者留下两个练习：

1. 如果  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  构成高斯随机向量，那么  <img src="https://www.zhihu.com/equation?tex=Y = AX + \bold b" alt="Y = AX + \bold b" class="ee_img tr_noresize" eeimg="1">  也是高斯随机向量吗？

2. 如果  <img src="https://www.zhihu.com/equation?tex=(Z1, Z2)^T" alt="(Z1, Z2)^T" class="ee_img tr_noresize" eeimg="1">  构成高斯随机向量，那么  <img src="https://www.zhihu.com/equation?tex=Z1 | Z2 = z" alt="Z1 | Z2 = z" class="ee_img tr_noresize" eeimg="1">  服从正态分布吗？

# 参考资料

[1] Jointly Normal Random Variables https://prob140.org/textbook/content/Chapter_23/00_Multivariate_Normal_RVs.html

[2] Multivariate Normal Distribution https://en.wikipedia.org/wiki/Multivariate_normal_distribution

[3] GaussianRandomVectors https://www.math.utah.edu/~davar/math6010/2014/GaussianRandomVectors.pdf
