# Reading materials
- [Reading materials](#reading-materials)
====

- [Recurrent Entity Networks with Delayed Memory Update for Targeted Aspect-based Sentiment Analysis](https://arxiv.org/abs/1804.11019)
  
    根据给定的文本数据特点对 Entity Network 记忆加入延迟更新

- [Deep Convolutional Neural Networks for Pairwise Causality](https://arxiv.org/abs/1701.00597)
  
    将数据对绘制成散点图，之后用 DCNN 分类： X->Y, 其他, Y->X

- [Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics](https://arxiv.org/abs/1706.04317)

    对环境中所有实体-属性之间的二值因果关系建模，在模型上通过搜索回溯规划动作

- [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)

    使用 Kronecker 分解二阶近似损失的曲率（ KL 散度）并使用信任区约束的参数更新改进 actor-critic 方法

- [Towards Deep Symbolic Reinforcement Learning](https://arxiv.org/abs/1609.05518)

    神经网络将高维环境状态压缩到低维概率一阶逻辑表示，以距离、类别、邻近信息作为概率特征，符号表示方法可能应用到文本实体解析中

- [Learning to Reinforcement Learn](https://arxiv.org/abs/1611.05763)

    RNN 元强化学习，我对文中展现的 RNN 抽象能力存疑，可能其模型结构正好适应其实验测试，而不是具备通用的泛化能力

- [Learning to perform physics experiments via deep reinforcement learning](https://arxiv.org/abs/1611.01843)

    深度强化模型从模拟中学习探索规律，通过对动作和目标代价估计平衡探索-利用，输入观测，输出因果解释的路线值得关注

- [RL2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779)

    使用大量采样训练通用强化学习算法，以 RNN 表示，作为小样本 MDP 问题的策略函数， RNN 隐状态作为小问题上的学习经验

- [High-Dimensional Continuous Control using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

    类似 TD(λ) 的一般优势估计，引入 λ 参数指数平滑 k 步优势估计， 0<λ<1 ， λ 小则估计方差小偏差大，反之同理；同时，用好的值函数近似变换远程激励为即时激励；优势估计也可以通过优化约束到策略的特征空间上

- [The History Began from AlexNet: A Comprehensive Survey on Deep Learning Approaches](https://arxiv.org/abs/1803.01164)

    深度学习综述，包括方法、计算效率、现成工具与资源等

- [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)

    深度强化学习的主要原理和改进方向，包括深度结构的使用如值迭代网络，生成对抗模仿学习，伪计数；改进估计稳定性的手段如对偶Q网络，确定性策略梯度，引导梯度搜索

- [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)

    深度强化学习的主要算法，包括深度 Q 学习，信任区策略优化，异步优势评价；常见技巧包括经验回放，目标网络等

- [A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity](https://arxiv.org/abs/1707.09183)

    非稳态环境造成了多 agent 学习的困难；待解决的挑战包括异构主体、大数量主体、动态交互、探索噪音等；理论性质包括均衡收敛性、最优策略收敛性、 regret 界、采样效率、鲁棒性等

- [Universal Reinforcement Learning Algorithms: Survey and Experiments](https://arxiv.org/abs/1705.10557)

    部分可观测 Markov 决策过程的一般强化学习框架，学习环境模型的后验概率，通过信息增益与效用边界权衡 exploration-exploitation

- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)

    采用中心化基线优势评价的多 agent 强化学习

- 网络表征学习前沿与实践

    保持网络拓扑空间中丰富的结构和属性是网络表征学习中的基本问题

- 网络表征学习中的基本问题初探

    高级信息保持的网络表征学习通常包括两个部分，一部分是使得节点表征保持原有网络结构信息，另一部分是构建起节点表征和目标任务之间的联系

- 异质网络表征学习的研究进展

    代表性的同质网络表征学习模型包括 DeepWalk, LINE, SDNE ；异质网络表征学习大致可以分为基于随机游走、分解和深度网络

- 属性网络表征学习的研究与发展

    大规模属性网络表征学习架构 AANE 根据每个节点的属性信息，计算出节点与节点之间的相似度矩阵或谱嵌入，进行对称分解映射至低维表示；同时驱使相连的节点拥有相似的低维向量，来求取联合的向量表示；将整个优化过程分解为 2n个独立子问题。
    属性网络表征学习中很多研究课题有待探索；如运用并行计算、随机梯度下降、负采样、局部优化等进一步提高表示学习的效率；整合更多类型的信息提升联合低维向量的性能；结合相关领域知识，设计问题来询问领域专家，答案被建模为新的链接加入原网络结构

- 全图表征学习的研究进展

    全图表示的目标是将整个网络表示成一个低维向量；深度图核函数将图分解为一系列子图结构作为词，整个子图结构集合对应一个句子，用词向量模型学习；基于卷积网络方法选择一些重要节点将其局部结构作为感受野的输入；神经消息通信算法用向量表示消息，图表示成为所有节点状态总和

- 人工智能的未来——记忆、知识、语言

    介绍了一些神经查询模型，符号查询器是自然语言到查询程序的 RNN ，强化学习，训练效率不高；神经查询器是问句的表示到符号实体的FNN，学习时首先训练神经查询器，以其结果训练符号查询器，问答时只使用符号查询器

- 大规模知识图谱研究及应用

    如“ schema 可分支定义的扩展机制”、“基于语义空间变换的实体归一技术”、“多层知识计算算子群”、“基于知识图谱进行语义理解的子图关联技术”、“基于 DNN 的 Type 语境搭配预测”、“基于自启动随机游走技术计算并控制概念泛化相关性”等技术方法具有启发价值

- 社会网络计算的回顾与展望, 社交网络中的用户影响力研究, 在线社交媒体中信息传播的建模与预测

    综述社会网络计算尤其是信息传播的研究，可以应用与舆情或灾害建模与预测，但大规模网络分析存在计算复杂度困难

- 异质信息网络的研究现状和未来发展

    知识图谱构建、挖掘、检索需要的方法综述

- 现代几何学与计算机科学

    将黎曼流形的特征函数推广到图上的工作值得关注，另外可以探索同调与同伦群作为先验约束流形学习的方法

- 追求视觉智能：对超越目标识别的探索, 从对象识别到场景理解

    两篇文章均重点提出了视觉概念语义关系在计算视觉任务中的重要性，分别采用强化学习和隐因子分解方式建模

- [StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856)

    Facebook 开源的通用向量空间嵌入方法，对多模态人工智能很实用

- [Hierarchical Nonlinear Orthogonal Adaptive-Subspace Self-Organizing Map based Feature Extraction for Human Action Recognition](http://ir.ia.ac.cn/handle/173211/19734)

    用于视频中人物动作分类的无监督特征提取模型，使用 kernel 映射分解特征

- [Action Recognition with Improved Trajectories](https://hal.inria.fr/hal-00873267v2/document)

    从视频中提取人物动作轨迹的方法

- [Distributed Representations of Words and Phrases and their Compositionality Efficient Estimation of Word Representations in Vector Space](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

    经典的词向量表示方法

- [Cortical microcircuits as gated-recurrent neural networks](https://arxiv.org/abs/1711.02448)

    隐状态抑制（减）激活的 LSTM

- [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)

    一个无监督文本段落表示模型，通过一句话来预测这句话的上一句和下一句

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

    新的激活函数，根据不动点定理保持均值为0，方差为1的合同映射，这篇文章证明了有界和收敛性质

- [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)

    利用知识库的问答系统

- [Marginalizing Corrupted Features](https://arxiv.org/abs/1402.7001)

    模型稳定性

- [Automatic Differentiation Variational Inference](https://arxiv.org/abs/1603.00788)

    快速变分推断方法

- [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://arxiv.org/abs/1111.4246)

    自适应采样路径长度的 Monte Carlo 采样

- [End-To-End Memory Networks](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)

    记忆网络模型通过对上下文集合和问题向量的变换，得到对应于问题的答案

- [Discriminative Embeddings of Latent Variable Models](https://arxiv.org/abs/1603.05629)

    将隐变量模型嵌入到向量空间的 kernel 方法

- [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)

    流生成网络，使用设计的 1x1 卷积等可逆变换，每一层输出一个量作为输入的编码

- [Learning Explanatory Rules from Noisy Data](https://arxiv.org/abs/1711.04574)

    将归纳逻辑命题描述为可微的神经网络模型，一定长度范围内所有可能的命题都通过网络模型并贡献了损失

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

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

    CycleGAN，使用对抗损失拟合互逆映射 $F(G(X))\approx X$ 与 $G(F(Y))\approx Y$
