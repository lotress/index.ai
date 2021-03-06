# Reading materials
- [Reading materials](#reading-materials)
  - [Theory](#theory)
  - [Modeling](#modeling)
  - [Transformation](#transformation)
  - [Decision](#decision)
  - [Memory](#memory)
  - [Optimization](#optimization)

====

## Theory

- 现代几何学与计算机科学

    将黎曼流形的特征函数推广到图上的工作值得关注，另外可以探索同调与同伦群作为先验约束流形学习的方法

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

    新的激活函数，根据不动点定理保持均值为0，方差为1的合同映射，这篇文章证明了有界和收敛性质

- [Marginalizing Corrupted Features](https://arxiv.org/abs/1402.7001)

    模型稳定性

- [Automatic Differentiation Variational Inference](https://arxiv.org/abs/1603.00788)

    快速变分推断方法

- [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://arxiv.org/abs/1111.4246)

    自适应采样路径长度的 Monte Carlo 采样

- [Nonlinear random matrix theory for deep learning](https://papers.nips.cc/paper/6857-nonlinear-random-matrix-theory-for-deep-learning.pdf)

    分析了激活矩阵 $M\equiv\frac{1}{m}Y^TY, Y=f(WX)$ 的迹，定义了与非线性激活函数相关的两个量 $\eta: f^2$ 的高斯平均， $\zeta: f'$ 高斯平均的平方，指出当 $\eta/\zeta$ 较大时训练损失较小

- [The Information-Autoencoding Family A Lagrangian Perspective On Latent Variable Generative Modeling](http://bayesiandeeplearning.org/2017/papers/60.pdf)

    指出一系列概率生成模型包括变分自编码器和对抗生成网络的学习目标是同一个原优化问题的 *Lagrangian* 对偶，该目标如下：
    $$
    \mathcal{L}(\boldsymbol{\lambda},\alpha_1,\alpha_2)=\alpha_1I_p(x;z)+\alpha_2I_q(x;z)+\sum^n_{i=1}\lambda_i\mathscr{D}_i
    $$
    其中 $\alpha_1,\alpha_2,\lambda_i\in\mathbb{R},\lambda_i>0$ 为 *Lagrangian* 乘子， $\mathscr{D}_i$ 项为下列三个集合之一的元素
    $$
    \{D(p(x,z)\|q(x,z))\}\\
    \{D(p(x)\|q(x)),\mathbb{E}_{r(x)}[D'(p(z|x)\|q(z|x))]\}\\
    \{D(p(z)\|q(z)),\mathbb{E}_{t(z)}[D'(p(x|z)\|q(x|z))]\}\\
    $$
    $D, D'$ 为 *KL* 与其反向散度， $r(x)$ 取 $p(x)$ 或 $q(x)$ ， $t(z)$ 取 $p(z)$ 或 $q(z)$

- [Uncovering divergent linguistic information in word embeddings with lessons for intrinsic and extrinsic evaluation](https://arxiv.org/abs/1809.02094)

    讨论了词向量的高阶相似度，设词向量矩阵 $X$ ，定义 n 阶相似度矩阵 $M_n(X)=(XX^T)^n$ ， $X^TX$ 的特征值分解为 $Q\Lambda Q^T$ ，令线性变换 $W_\alpha=Q\Lambda^\alpha$ ，可知 $M_n(X)=M_1(XW_{\frac{n-1}{2}})$ ，故调整实数域上的参数 $\alpha$ 即可表示词向量的高阶相似度

- [Global versus Localized Generative Adversarial Nets](https://arxiv.org/abs/1711.06020)

    提出以数据流形上局部切空间的正交基作为生成模型的表示，生成器为数据点 $x$ 与局部坐标 $z$ 的函数 $G(x,z)=x$ ，多个局部生成器拼合整个数据流形。其中局部性约束 $G(x,0)=x$ 使得平直空间中的 $z$ 的改变在 $x$ 的小邻域上可以直接映射为生成数据的属性变化，正交性约束 $J_x^TJ_x=I_N$ 防止生成器模式塌缩， *Jacobian* 矩阵 $J_x\triangleq\frac{\partial G(x,z)}{\partial z}|_{z=0}\in\mathbb{R}^{D\times N}$ ，上述两约束的正则化项可加入生成器的损失中训练。判别器 $P(y|x)$ 可与 $K$ 类分类任务联合半监督训练，其目标为
    $$
    \underset{P}{max}\mathbb{E}_{(x_l,y_l)\sim P_{\mathcal{L}}}\log P(y_l|x_l)+\mathbb{E}_{x_u\sim P_{\mathcal{X}}}\log P(y_u\leq K|x_u)+\mathbb{E}_{x\sim P_{\mathcal{X}},z\sim P_{\mathcal{Z}}}\log P(y=K+1|G(x,z))-\sum^K_{k=1}\mathbb{E}_{x\sim P_{\mathcal{X}}}\|\nabla_x^G\log P(y=k|x)\|^2
    $$
    其中 $(x_l,y_l)$ 为 $K$ 类标记数据， $x_u$ 为非标记数据， $K+1$ 类为生成数据， $\nabla_x^Gf\triangleq\nabla_zf(G(x,z))|_{z=0}=J_x^T\nabla_xf(x)$ ，最后一项惩罚分类器 $f$ 在流形切平面上的突然变化， $|f(G(x,z+\delta z))-f(G(x,z))|^2\approx\|\nabla_x^Gf\|^2\delta z$ 。

- [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)

    分析图神经网络的表示能力，一结点上的一次迭代将（特征）标签的multiset映射到新的（特征）标签，故表示能力最大的图神经网络其（必要非充分条件）结点与全图的聚合映射必然为单射方可区分不同的输入multiset，此时该变换与Weisfeiler-Lehma图同构判定过程等价；累积和就是个满足单射条件的聚合映射。

- [Towards A Deep and Unified Understanding of Deep Neural Models in NLP](https://www.microsoft.com/en-us/research/publication/towards-a-deep-and-unified-understanding-of-deep-neural-models-in-nlp/)

    选择了互信息量化神经网络中间状态编码的容量，对于n个词的句子X和其中间状态表示$\Phi(x)$学习一组球形正态噪声$\epsilon=[\epsilon_1^T, \epsilon_2^T, \dots, \epsilon_n^T]^T$扰动$\tilde{x}_i=x_i+\epsilon_i$，最小化损失$L(\sigma)=E_{\epsilon}\|\Phi(\tilde{x})-\Phi(x)\|^2-\lambda\sum_{i=1}^{n}H(\tilde{X}_i|\Phi(x))|_{\epsilon_i\sim N(0,\sigma_i^2I)}$，其中$\lambda$为大于0的超参数，期望项最大化$\sum_{i}\sum_{\tilde{x}_i}\log p(\tilde{x}_i|\Phi(x))$，松弛项鼓励更大的条件熵，即该噪声应尽可能扰动输入而保持状态表示不变。注意关于噪声的期望拟合了表示的条件分布，所以可以用$H(\tilde{X}_i|\Phi(x))$近似$H(X_i|\Phi(x))$，即$p(\tilde{x}_i|\Phi(x))\approx p(x_i|\Phi(x))\Rightarrow H(X_i|\Phi(x))\approx H(\tilde{X}_i|\Phi(x))=\frac{K}{2}\log(2\pi e)+K\log(\sigma_i)$

- [Group Representation Theory for Knowledge Graph Embedding](https://grlearning.github.io/papers/15.pdf)

    三元组的关系可视为群的元素，交换群的任意有限维不可约复表示都是一维的，即群 G 的一个不可约表示是同态 $G\rightarrow GL_1(\mathbb{C})=\mathbb{C}^*$ ， $\mathbb{C}^*$ 为一个非平凡复乘法群；对有限交换群，不可约表示是单位根的数乘，所以复平面上的旋转可表示任意有限交换群；非交换群的表示可通过将复数域扩展为四元数域建模

- [Logical Expressiveness of Graph Neural Networks](https://grlearning.github.io/papers/92.pdf)

    以 Weisfeiler-Lehman 图同构测试论证典型的领域聚合图网络（AC-GNN）表示能力弱于 $FOC_2$ （带计数量词的2变量一阶逻辑）， 而 AC-GNN 包含描述逻辑 ALCQ ，其禁止了 $FOC_2$ 的三种非局域公式 1) 否定二元关系如 $\neg E(x,y)$ 2) 无自由变量的子式如 $R(v)\wedge\exist{x}B(x)$ 3) 不连接的公式如 $\exist{v}(R(v)\wedge B(x))$ ；当 GNN 每次聚合时加入图的全局表示信息就等价于一元 $FOC_2$

## Modeling

- [The History Began from AlexNet: A Comprehensive Survey on Deep Learning Approaches](https://arxiv.org/abs/1803.01164)

    深度学习综述，包括方法、计算效率、现成工具与资源等

- [Deep Convolutional Neural Networks for Pairwise Causality](https://arxiv.org/abs/1701.00597)
  
    将数据对绘制成散点图，之后用 DCNN 分类： X->Y, 其他, Y->X

- [Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics](https://arxiv.org/abs/1706.04317)

    对环境中所有实体-属性之间的二值因果关系建模，在模型上通过搜索回溯规划动作

- 人工智能的未来——记忆、知识、语言

    介绍了一些神经查询模型，符号查询器是自然语言到查询程序的 RNN ，强化学习，训练效率不高；神经查询器是问句的表示到符号实体的FNN，学习时首先训练神经查询器，以其结果训练符号查询器，问答时只使用符号查询器

- 大规模知识图谱研究及应用

    如“ schema 可分支定义的扩展机制”、“基于语义空间变换的实体归一技术”、“多层知识计算算子群”、“基于知识图谱进行语义理解的子图关联技术”、“基于 DNN 的 Type 语境搭配预测”、“基于自启动随机游走技术计算并控制概念泛化相关性”等技术方法具有启发价值

- 社会网络计算的回顾与展望, 社交网络中的用户影响力研究, 在线社交媒体中信息传播的建模与预测

    综述社会网络计算尤其是信息传播的研究，可以应用与舆情或灾害建模与预测，但大规模网络分析存在计算复杂度困难

- 异质信息网络的研究现状和未来发展

    知识图谱构建、挖掘、检索需要的方法综述

- 属性网络表征学习的研究与发展

    大规模属性网络表征学习架构 AANE 根据每个节点的属性信息，计算出节点与节点之间的相似度矩阵或谱嵌入，进行对称分解映射至低维表示；同时驱使相连的节点拥有相似的低维向量，来求取联合的向量表示；将整个优化过程分解为 2n个独立子问题。
    属性网络表征学习中很多研究课题有待探索；如运用并行计算、随机梯度下降、负采样、局部优化等进一步提高表示学习的效率；整合更多类型的信息提升联合低维向量的性能；结合相关领域知识，设计问题来询问领域专家，答案被建模为新的链接加入原网络结构

- 追求视觉智能：对超越目标识别的探索, 从对象识别到场景理解

    两篇文章均重点提出了视觉概念语义关系在计算视觉任务中的重要性，分别采用强化学习和隐因子分解方式建模

- [Hierarchical Nonlinear Orthogonal Adaptive-Subspace Self-Organizing Map based Feature Extraction for Human Action Recognition](http://ir.ia.ac.cn/handle/173211/19734)

    用于视频中人物动作分类的无监督特征提取模型，使用 kernel 映射分解特征

- [Action Recognition with Improved Trajectories](https://hal.inria.fr/hal-00873267v2/document)

    从视频中提取人物动作轨迹的方法

- [Cortical microcircuits as gated-recurrent neural networks](https://arxiv.org/abs/1711.02448)

    隐状态抑制（减）激活的 LSTM

- [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)

    利用知识库的问答系统

- [Learning Explanatory Rules from Noisy Data](https://arxiv.org/abs/1711.04574)

    将归纳逻辑命题描述为可微的神经网络模型，一定长度范围内所有可能的命题都通过网络模型并贡献了损失

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

    CycleGAN，使用对抗损失拟合互逆映射 $F(G(X))\approx X$ 与 $G(F(Y))\approx Y$

- [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)

    认为组合泛化是重要目标而结构表示对此很关键，描述了偏好关系归纳的图网络框架，作为学习与系统中实体数量和顺序无关泛化的一般范式

- [How Can We Be So Dense? The Benefits of Using Highly Sparse Representations](https://arxiv.org/abs/1903.11257)

    认为稀疏表示有利于抵抗噪音，两各维度独立分布的随机向量内积相似度大于一定阈值的概率，相对向量非零分量与总维度比例呈指数关系。文章提出的结构仅激活固定数量的输出分量，初始化固定数量的矩阵权重，其余权重保持为0，激活函数为保留 top-k ，为均衡各分量的激活频率，以各分量的激活频率加权，推断时可以加大 k 。

- [Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)

    评估多个自监督图像表示模型和任务对下游任务的影响，结论包括：模型结构很重要，残差比VGG类的下游表现好；自监督效果好的模型未必下游表现好；越宽（通道数等）越好；下游用 SGD 训练 Logisitc 回归收敛很慢。

## Transformation

- [Towards Deep Symbolic Reinforcement Learning](https://arxiv.org/abs/1609.05518)

    神经网络将高维环境状态压缩到低维概率一阶逻辑表示，以距离、类别、邻近信息作为概率特征，符号表示方法可能应用到文本实体解析中

- 网络表征学习前沿与实践

    保持网络拓扑空间中丰富的结构和属性是网络表征学习中的基本问题

- 网络表征学习中的基本问题初探

    高级信息保持的网络表征学习通常包括两个部分，一部分是使得节点表征保持原有网络结构信息，另一部分是构建起节点表征和目标任务之间的联系

- 异质网络表征学习的研究进展

    代表性的同质网络表征学习模型包括 DeepWalk, LINE, SDNE ；异质网络表征学习大致可以分为基于随机游走、分解和深度网络

- 全图表征学习的研究进展

    全图表示的目标是将整个网络表示成一个低维向量；深度图核函数将图分解为一系列子图结构作为词，整个子图结构集合对应一个句子，用词向量模型学习；基于卷积网络方法选择一些重要节点将其局部结构作为感受野的输入；神经消息通信算法用向量表示消息，图表示成为所有节点状态总和

- [StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856)

    Facebook 开源的通用向量空间嵌入方法，对多模态人工智能很实用

- [Distributed Representations of Words and Phrases and their Compositionality Efficient Estimation of Word Representations in Vector Space](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

    经典的词向量表示方法

- [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)

    一个无监督文本段落表示模型，通过一句话来预测这句话的上一句和下一句

- [Discriminative Embeddings of Latent Variable Models](https://arxiv.org/abs/1603.05629)

    将隐变量模型嵌入到向量空间的 kernel 方法

- [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)

    流生成网络，使用设计的 1x1 卷积等可逆变换，每一层输出一个量作为输入的编码

- [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)

    讨论了平移不变的卷积的局限性，提出 `CoordConv` 即图像特征拼接位置特征再卷积的操作克服了位置局限性并具有广泛应用前景

- [Predicting the Generalization Gap in Deep Networks with Margin Distributions](https://arxiv.org/abs/1810.00113)

    提出神经网络层的归一化 margin 分布可以良好估计泛化 gap ，给定 l 层的特征表示 $x^l$ ，以一阶泰勒近似分类器 f 关于类别 i, j 决策边界的 margin ， $d_{f,(i,j)}(x^l)=\frac{f_i(x^l)-f_j(x^l)}{\|\nabla_{x^l} f_i(x^l)-\nabla_{x^l} f_j(x^l)\|_2}$ ，并以表示的协方差归一化得到 margin 分布 $\hat{d}_{f,(i,j)}(x_k^l)=\frac{d_{f,(i,j)}(x_k^l)}{\sqrt{v(x^l)}}\ ,\ v(x^l)=tr(\frac{1}{n}\sum_{k=1}^{n}(x_k^l-\hat{x}^l)(x_k^l-\hat{x}^l)^T)\ ,\ \hat{x}^l=\frac{1}{n}\sum_{k=1}^{n}x_k^l$ 。文献的实验使用 margin 分布的四分卫点等特征建立对数线性模型预测深度模型的泛化 gap 。

- [Hyperbolic Graph Convolutional Neural Networks](https://github.com/HazyResearch/hgcn)

    d 维 Minkowski 内积 $\langle x, y \rangle_\mathcal{L}=-x_0y_0+x_1y_1+\cdots +x_dy_d$ ，具有常负曲率 $-1/K(K>0)$ 的双曲面空间 $\mathbb{H}^{d,K}:=\{x\in \mathbb{R}^{d+1}:\langle x,x \rangle_\mathcal{L}=-K,x_0>0\}$ ，其切空间 $\mathcal{\tau}_x\mathbb{H}^{d,K}:=\{v\in\mathbb{R}^{d+1}:\langle v,x \rangle_\mathcal{L}=0\}$ 是双曲面流形在 x 处的一阶局部近似且其 Minkowski 内积正定，故在切空间中进行领域聚合等欧式距离的运算，双曲面流形上的测地距离则为 $d_\mathcal{L}^K(x,y)=\sqrt{K}\mathrm{arccosh}(-\langle x,y \rangle_\mathcal{L}/K)$ ，此距离能更好反映树/图中指数扩大的邻域；于是有双曲面流形到切空间映射 $\log_x^K(y)=d_\mathcal{L}^K(x,y)\frac{y+\frac{1}{K}\langle x,y \rangle_\mathcal{L}x}{\|y+\frac{1}{K}\langle x,y \rangle_\mathcal{L}x\|_\mathcal{L}}$ ，逆映射 $\exp_x^K(v)=\cosh(\frac{\|v\|_\mathcal{L}}{\sqrt{K}})x+\sqrt{K}\sinh(\frac{\|v\|_\mathcal{L}}{\sqrt{K}})\frac{v}{\|v\|_\mathcal{L}}$ ，向量 $v\in\mathbb{R}^{d+1}$ 在 $\mathcal{\tau}_x\mathbb{H}^{d,K}$ 上的投影 $\Pi(v):=v+\frac{1}{K}\langle v,x \rangle_\mathcal{L}x$ ，而切向量 $v\in\mathcal{\tau}_x\mathbb{H}^{d,K}$ 到切空间 $\mathcal{\tau}_y\mathbb{H}^{d,K}$ 的平移为 $P_{x\rightarrow y}(v)=v-\frac{\langle \log_x^K(y),v \rangle_\mathcal{L}}{d_\mathcal{L}^K(x,y)^2}(\log_x^K(y)+\log_y^K(x))$ ；非线性变换需要连接不同的空间曲率，因此在不受曲率影响的原点上映射双曲面流形与切空间 $\sigma^{\otimes
K_{\mathcal{L}-1},K_\mathcal{L}}=\exp_o^{K_\mathcal{L}}(\sigma(\log_x^{K_{\mathcal{L}-1}}(x^H)))$

- [A Geometric Understanding of Deep Learning](https://arxiv.org/abs/1805.10451)

  GAN的任务可分为流形学习和概率分布变换两部分，后者的求传输映射$T:X\rightarrow Y$的对偶问题是给定代价函数$c(x, y): X \times Y\rightarrow\mathbb{R}_{\geq 0}$求实函数$\varphi:X\rightarrow\mathbb{R}$和$\psi:Y\rightarrow\mathbb{R}$，$\max_{\varphi,\psi}[\int_{X}\varphi(x)\mathrm{d}\mu+\int_{Y}\psi(y)\mathrm{d}\nu:\varphi(x)+\psi(y)\le c(x, y)]$给出了Wasserstein距离，$\varphi$可称为Kantorovich势；令$c(x, y)=1/2\|x-y\|^2$，存在称为Brenier势的唯一凸函数$u:X\rightarrow\mathbb{R}$，使得$\mu$到$\nu$的传输映射$\varphi=\nabla u^\circ s$，s为$X\rightarrow X$的保测映射；对紧致凸域上的$\mu$和$\nu$，有$T(x)=x-\nabla\varphi(x)=\nabla[\frac{1}{2}\|x\|^2-\varphi(x)]=\nabla u(x)$；Brenier势的图像是个凸多面体，每个训练样本$y_i$对应Brenier势的一个支撑平面$\pi_{h,i}(x)=\langle x,y_i\rangle+h_i$，在$\mu$的支撑域$\Omega$上的投影$W_i(h)$构成$\Omega$的一个分解；Brenier势可表示为由$h$确定的分段线性函数$u_h(x)=\max_{i=1}^n[\pi_{h,i}(x)]=\max_{i=1}^n[\langle x,y_i\rangle+h_i]$；记$W_i$的$\mu$测度为$w_i(h)$，未知的高度变量$h$最小化凸能量$E(h)=\int_0^h\sum_{i=1}^{n}w_i(\eta)\mathrm{d}\eta_i-\sum_{i=1}^{n}h_i\nu_i$，可通过Monte Carlo方法估计$w_i(h)$从而规划求解最小化$E(h)$；两个类别的邻接边界产生了传输映射的不连续，通过两个样本间的角度$\theta_{i,j}=\frac{\langle y_i,y_j\rangle}{\|y_i\|\|y_j\|}$阈值判别；在上述代价函数下，$u$与$\varphi$关系为$u(x)=1/2\|x\|^2-\varphi(x)$，$\nabla u$给出了生成模型$(g_\theta)_\#\zeta$到真实分布$\nu$的最优传输映射；对参数$\theta$的第k次迭代的生成模型$g_\theta^k$和参数$\xi$的判别模型$\varphi_\xi$有$v=(\nabla u)_\#[(g_\theta^k)_\#\zeta]=(\nabla u\circ g_\theta^k)_\#\zeta=[(\mathrm{id}-\nabla_{\varphi_\xi})\circ g_\theta^k]_\#\zeta$；于是生成模型$g_\theta^{k+1}=(\mathrm{id}-\nabla_{\varphi_\xi})\circ g_\theta^k$

  ![Brenier’s potential](image.readings/A-Geometric-Understanding-of-Deep-Learning.Fig.6.png)

## Decision

- [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)

    深度强化学习的主要原理和改进方向，包括深度结构的使用如值迭代网络，生成对抗模仿学习，伪计数；改进估计稳定性的手段如对偶Q网络，确定性策略梯度，引导梯度搜索

- [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)

    深度强化学习的主要算法，包括深度 Q 学习，信任区策略优化，异步优势评价；常见技巧包括经验回放，目标网络等

- [A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity](https://arxiv.org/abs/1707.09183)

    非稳态环境造成了多 agent 学习的困难；待解决的挑战包括异构主体、大数量主体、动态交互、探索噪音等；理论性质包括均衡收敛性、最优策略收敛性、 regret 界、采样效率、鲁棒性等

- [Learning to Reinforcement Learn](https://arxiv.org/abs/1611.05763)

    RNN 元强化学习，我对文中展现的 RNN 抽象能力存疑，可能其模型结构正好适应其实验测试，而不是具备通用的泛化能力

- [Learning to perform physics experiments via deep reinforcement learning](https://arxiv.org/abs/1611.01843)

    深度强化模型从模拟中学习探索规律，通过对动作和目标代价估计平衡探索-利用，输入观测，输出因果解释的路线值得关注

- [RL2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779)

    使用大量采样训练通用强化学习算法，以 RNN 表示，作为小样本 MDP 问题的策略函数， RNN 隐状态作为小问题上的学习经验

- [High-Dimensional Continuous Control using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

    类似 TD(λ) 的一般优势估计，引入 λ 参数指数平滑 k 步优势估计， 0<λ<1 ， λ 小则估计方差小偏差大，反之同理；同时，用好的值函数近似变换远程激励为即时激励；优势估计也可以通过优化约束到策略的特征空间上

- [Universal Reinforcement Learning Algorithms: Survey and Experiments](https://arxiv.org/abs/1705.10557)

    部分可观测 Markov 决策过程的一般强化学习框架，学习环境模型的后验概率，通过信息增益与效用边界权衡 exploration-exploitation

- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)

    采用中心化基线优势评价的多 agent 强化学习

## Memory

- [Tracking The World State With Recurrent Entity Networks](https://arxiv.org/abs/1612.03969)

    Recurrent Entity Network ，通过内容寻址捕捉输入从而对记忆单元进行实时的更新

- [Recurrent Entity Networks with Delayed Memory Update for Targeted Aspect-based Sentiment Analysis](https://arxiv.org/abs/1804.11019)
  
    根据给定的文本数据特点对 Entity Network 记忆加入延迟更新

- [End-To-End Memory Networks](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)

    记忆网络模型通过对上下文集合和问题向量的变换，得到对应于问题的答案

## Optimization

- [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)

    使用 Kronecker 分解二阶近似损失的曲率（ KL 散度）并使用信任区约束的参数更新改进 actor-critic 方法

- [Welcoming the Era of Deep Neuroevolution](https://eng.uber.com/deep-neuroevolution/)

    综述使用进化算法训练神经网络，对比进化算法与梯度方法，进化算法更容易跨越梯度障碍，不容易进入狭窄但连续的梯度方向；根据参数敏感程度即模型输出关于参数的梯度成比例变异更加稳定；纳入多样性/新颖性的群体进化算法在激励稀疏的问题上可能效果更好

- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)

    机器学习系统设计相关的风险因素，包括系统边界侵蚀、纠缠（多个模型间的非单调关系）、不可见反馈、不可见消费者、数据依赖、配置问题、外部环境变化和若干反模式

- [The Perception-Distortion Tradeoff](https://arxiv.org/abs/1711.06077)

    证明图像还原任务的失真损失$\mathbb{E}[\Delta(X,\hat{X})]$与感知损失$d(p_X,p_{\hat{X}|Y})$不可兼得，其中X为原图像，$\hat{X}$为给定输入Y下的还原输出，$d(p,q)$为关于q的凸概率散度，则感知-失真损失函数$P(D)=\min_{p_{\hat{X}|Y}}d(p_X,p_{\hat{X}})\ s.t.\ \mathbb{E}[\Delta(X,\hat{X})]\le D$为单调凸函数；文献建议共同使用侧重失真的full-reference与侧重感知的no-reference指标评估图像还原任务。

- [Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801)

    将随机梯度下降的参数更新和动量衰减因子解耦，减少动量估计引入的偏差， QHM 更新规则如下
    $$
    \begin{aligned}
    g_{t+1}&\leftarrow\beta\cdot g_{t}+(1-\beta)\cdot\nabla\hat{L}_t(\theta_t)\\
    \theta_{t+1}&\leftarrow\theta_t-\alpha[(1-\nu)\cdot\nabla\hat{L}_t(\theta_t)+\nu\cdot g_{t+1}]
    \end{aligned}
    $$
    类似改动 Adam 优化，QHAdam 更新规则如下
    $$
    \begin{aligned}
    g_{t+1}&\leftarrow\beta_1\cdot g_{t}+(1-\beta_1)\cdot\nabla\hat{L}_t(\theta_t) &g'_{t+1}\leftarrow(1-\beta_1^{t+1})^{-1}\cdot g_{t+1}\\
    s_{t+1}&\leftarrow\beta_2\cdot s_{t}+(1-\beta_2)(\nabla\hat{L}_t(\theta_t))^2 &s'_{t+1}\leftarrow(1-\beta_2^{t+1})^{-1}\cdot s_{t+1}\\
    \theta_{t+1}&\leftarrow\theta_t-\alpha[\frac{(1-\nu_1)\cdot\nabla\hat{L}_t(\theta_t)+\nu_1\cdot g'_{t+1}}{\sqrt{(1-\nu_2)(\nabla\hat{L}_t(\theta_t))^2+\nu_2\cdot s'_{t+1}}+\epsilon}]
    \end{aligned}
    $$