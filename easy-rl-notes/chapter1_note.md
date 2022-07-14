# 1 概述

## 1.1 介绍

- 强化学习讨论的问题是一个智能体（agent）怎么在一个复杂不确定的环境（environment）里面去极大化它能获得的奖励。

- 监督学习与强化学习对比：
  - 监督学习数据独立同分布；有标签。
  - 强化学习是时间序列，不满足独立同分布；没有立刻获得反馈，面临延迟奖励。
- 强化学习可能有超人类表现，在环境中自行探索；监督学习标签只能由人类标注，无法超越人类。

- 一场游戏叫做一个episode（回合）或trial（实验）。
- 将神经网络放入强化学习的方法：
  - standard RL：设计特征训练分类网络作出动作决策。
  - Deep RL：端到端的输出动作，省去手工设置动作过程。
- 强化学习应用增多原因：
  - 更高的算力。
  - 通过不同尝试获得更多信息。
  - 端到端训练形成更强决策网络。

## 1.2 序列决策过程

- agent目的是极大化它的期望的累积奖励。

- 状态和观测的关系：
  - 状态不会隐藏世界信息，观测会遗漏一些信息。
  - 状态和观测完全等价时，强化学习可以被建模为MDP问题。
  - 当agent只知道环境的部分信息时，强化学习通常被建模为部分可观测马尔科夫决策过程（POMDP）。POMDP可以用一个7元组（S，A，T，R，$\Omega$，$O$，$\gamma$）表示。

## 1.3 动作空间

- 离散动作空间的动作是有限的，连续动作空间的动作是实值向量。

## 1.4 RL Agent的主要组件

- 强化学习Agent主要由策略函数（policy）、价值函数（value）、模型组成（model）。
- **Policy**是agent的行为模型，决定了agent的行为：
  - stochastic policy（随机行策略）：指$\pi(a|s) = P[A_t=a|S_t=s]$，输出概率，然后对概率分布进行采样获得动作分布。
  - deterministic policy（确定性策略）：采取最有可能的动作。
- 通常强化学习更多采用随机性策略，引入随机性更好地探索环境；多智能体博弈时，采用确定性的策略智能体总是对环境做出相同动作，导致他的策略容易被预测。
- **价值函数**是未来奖励的一个预测，用来评估状态的好坏，分为价值函数（自变量为S）和Q函数（自变量为S、A）。

$$
\mathrm{v}_{\pi}(\mathrm{s}) \doteq \mathbb{E}_{\pi}\left[\mathrm{G}_{\mathrm{t}} \mid \mathrm{S}_{\mathrm{t}}=\mathrm{s}\right]=\mathbb{E}_{\pi}\left[\sum_{\mathrm{k}=0}^{\infty} \mathrm{\gamma}^{\mathrm{k}} \mathrm{R}_{\mathrm{t}+\mathrm{k}+1} \mid \mathrm{S}_{\mathrm{t}}=\mathrm{s}\right], \text { for all } \mathrm{s} \in \mathcal{S}\\
\mathrm{q}_{\pi}(\mathrm{s}, \mathrm{a}) \doteq \mathbb{E}_{\pi}\left[\mathrm{G}_{\mathrm{t}} \mid \mathrm{S}_{\mathrm{t}}=\mathrm{s}, \mathrm{A}_{\mathrm{t}}=\mathrm{a}\right]=\mathbb{E}_{\pi}\left[\sum_{\mathrm{k}=0}^{\infty} \mathrm{\gamma}^{\mathrm{k}} \mathrm{R}_{\mathrm{t}+\mathrm{k}+1} \mid \mathrm{S}_{\mathrm{t}}=\mathrm{s}, \mathrm{A}_{\mathrm{t}}=\mathrm{a}\right]
$$

- **模型**决定了下一个状态是什么，取决于当前状态和当前行为，由两个部分组成：

  - 概率：状态间转移概率

  $$
  \mathcal{P}_{\mathrm{ss}^{\prime}}^{\mathrm{a}}=\mathbb{P}\left[\mathrm{S}_{\mathrm{t}+1}=\mathrm{s}^{\prime} \mid \mathrm{S}_{\mathrm{t}}=\mathrm{s}, \mathbf{A}_{\mathrm{t}}=\mathrm{a}\right]
  $$

  - 奖励函数：当前状态下行为的奖励
    $$
    \mathcal{R}_{\mathrm{s}}^{\mathrm{a}}=\mathbb{E}\left[\mathrm{R}_{\mathrm{t}+1} \mid \mathrm{S}_{\mathrm{t}}=\mathrm{s}, \mathrm{A}_{\mathrm{t}}=\mathrm{a}\right]
    $$

## 1.5 RL Agents的种类

- 根据学习内容方法分类

  - 基于价值的agent，显示学习价值函数，推断出最优策略。

  - 基于策略的agent，学习动作概率，直接学习policy。

  - 结合价值和策略，形成actor critic agent，交互获得最佳行为。

- 基于价值迭代的强化学习算法有 Q-learning、 Sarsa 等，而基于策略迭代的强化学习算法有策略梯度算法等。
- 根据学习环境分类
  - model-based，它通过学习这个状态的转移来采取动作。
  - model-free，它没有去直接估计这个状态的转移，也没有得到环境的具体转移变量。它通过学习价值函数和策略函数进行决策。Model-free 的模型里面没有一个环境转移的模型。

- 通常情况下，状态转移函数和奖励函数很难估计，甚至连环境中的状态都可能是未知的，这时就需要采用免模型学习。 免模型学习没有对真实环境进行建模，智能体只能在真实环境中通过一定的策略来执行动作，等待奖励和状态迁移，然后根据这些反馈信息来更新行为策略，这样反复迭代直到学习到最优策略。在实际应用中，如果不清楚该用有模型强化学习还是免模型强化学习，可以先思考一下，在智能体执行动作前，是否能对下一步的状态和奖励进行预测，如果可以，就能够对环境进行建模，从而采用有模型学习。
- 有模型学习可以在一定程度上缓解训练数据匮乏的问题，因为智能体可以在虚拟世界中行训练。
- 免模型学习的泛化性要优于有模型学习。

- 有模型的强化学习方法可以对环境建模，使得该类方法具有独特魅力，即“想象能力”。在免模型学习中，智能体只能一步一步地采取策略，等待真实环境的反馈；而有模型学习可以在虚拟世界中预测出所有将要发生的事，并采取对自己最有利的策略。

- 目前，大部分深度强化学习方法都采用了免模型学习，原因有：
  - 免模型学习更为简单直观且有丰富的开源资料；
  - 在目前的强化学习研究中，大部分情况下环境都是静态的、可描述的，智能体的状态是离散的、可观察的（如 Atari 游戏平台），这种相对简单确定的问题并不需要评估状态转移函数和奖励函数，直接采用免模型学习，使用大量的样本进行训练就能获得较好的效果。

## 1.6 Exploration and Exploitation

- 探索是指尝试不同行为获得最大化策略。
- 利用是指不尝试新的东西通过已知策略获得最大化奖励。

## 1.7 实验

- 学术界一般最关心一百回合的平均回合奖励。