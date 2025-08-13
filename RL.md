# Reinforcement learning models timeline and key papers

- **2021 arXiv(CoRL 2021): “Implicit Behavioral Cloning” (Florence et al.)**

  > [Paper](https://arxiv.org/abs/2109.00137) & [Openreview](https://openreview.net/forum?id=rif3a5NAxU6) & [Website](https://implicitbc.github.io/) & [Video](https://www.youtube.com/watch?v=QslGqRUSRzs) & [Code](https://github.com/google-research/ibc)

   Behavior cloning is arguably the simplest possible way to get a policy. Given a set of expert observations and actions $\{o_i, a_i\}$, we learn a conditional generative model that maps from observations to actions, $a\sim p(a\|o)$. The authors show that implicit behavioral cloning policies with energy-based models (EBM) often outperform common explicit (Mean Square Error, or Mixture Density) behavioral cloning policies on robotic policy learning tasks, including on tasks with high-dimensional action spaces and visual image inputs.


  In this work, they propose to reformulate behavior cloning (BC) using implicit models — specifically, the composition of `argmin` with a continuous energy function $E_\theta$ to represent the policy $\pi_\theta$: $\hat{a} = \arg\min_{a \in \mathcal{A}} E_\theta(o, a)$ instead of $\hat{a} = F_\theta(o)$.
