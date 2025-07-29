# diffusion‑model timeline and key papers
> [LiL's Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
## 1. Foundational Theory and Early Pioneering Works (2015–2019)

- **2015 ICML: “Deep Unsupervised Learning using Nonequilibrium Thermodynamics” (Sohl‑Dickstein et al.)**

  > [Paper](https://arxiv.org/abs/1503.03585) & [Video](https://www.youtube.com/watch?v=XLzhbXeK-Os) & [Code](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/tree/master)

  The authors first propose a diffusion‑model framework. Inspired by non-equilibrium statistical physics, they define a forward diffusion process which converts any complex data distribution into a simple, analytically tractable distribution (such as a zero-mean, unit-covariance Gaussian) and then trains a neural network to learn a finite-time reversal of this diffusion process which deifnes generative model distribution.

  ![Figure 1. The proposed modeling framework trained on 2-d swiss roll data.](./assets/figure1.png)
  
  **Derivation of the Evidence Lower Bound (ELBO)**: Converting likelihood maximization into log‑likelihood maximization $\mathcal L=\mathbb{E}_{q(x^{(0)})}[\log p(x^{(0)})]$, so that Jensen’s inequality can turn $\log\int$ into a computable lower bound of $\int\log$, then splitting that bound across time steps so that each term is a KL divergence.  

   **Optimization objective & training**: By treating each reverse diffusion kernel as a parametric model, the core training objective becomes finding the optimal parameters of mean and covariance functions of each step’s reverse kernel that maximize this log‑likelihood bound, which is equivalent to simultaneously minimizing the KL divergence between the reverse kernel at each step and the true posterior. In this way, estimating a complex distribution reduces to predicting the parameters needed for each reverse diffusion step.

- **2019 NeurIPS: “Generative Modeling by Estimating Gradients of the Data Distribution” (YSong & Ermon)**
  > [Paper](https://arxiv.org/abs/1907.05600) & [Blog](http://yang-song.net/blog/2021/score/) & [Video](https://www.youtube.com/watch?v=8TcNXi3A5DI) & [Code](https://github.com/ermongroup/ncsn) & [Summary Video](https://www.youtube.com/watch?v=wMmqCMwuM2Q)  
  
  The authors propose the framework of score-based generative modeling where they first estimate gradient of data log‑density, $\nabla_x \log p_{\rm data}(x)$, via score matching, and then generate samples by iteratively taking small steps in the direction of this learned score while injecting noise via Langevin dynamics. In this way, random noise “climbs” up the learned log‑density landscape into high‑probability regions, producing realistic new samples.

  ![Figure 2. The proposed score-based modeling framework with score matching and Langevin dynamics.](./assets/figure2.png)

  **Working with score functions**: Unlike the statistical score function, the score in score matching is the gradient with respect to the input $x$, not the gradient with respect to the model parameters $\theta$. Here, the score function is the vector field that gives the direction where the density function grows most quickly.

  **The key idea of score-based modeling framework**: Note that Langevin dynamics can produce samples from a probability density $p(\mathbb x)$ using only the score function $\nabla_{\mathbb x} \log p(\mathbb x)$. In order to obtain samples from $p_\text{data}(\mathbb x)$, first train score network such that $\mathbb s_\theta(\mathbb x) \approx \nabla_x \log p_\text{data}(\mathbb x)$ and then approximately obtain samples with Langevin dynamics using $\mathbb s_\theta(\mathbb x)$.

  ![Figure 3. The improved score-based modeling framework with denosing score matching and annealed Langevin dynamics.](./assets/figure3.png)

  **Improved score-based generative modeling**: Based on the observation that perturbing data with random Gaussian noise makes the distribution more amenable to score‑based generative modeling,  first corrupt the data at multiple noise levels and then train a Noise Conditional Score Network (NCSN), $s_\theta(x,\sigma)\approx\nabla_x\log q_\sigma(x)$ to jointly estimate the scores for all noise scales. This network combines a U‑Net architecture with dilated convolutions and employs instance normalization. Once the NCSN $s_\theta(x,\sigma)$ is trained, inspired by simulated annealing and annealed importance sampling, they propose a sampling procedure—annealed Langevin dynamics, because the rough intuition is they hope to gradually anneal down the temperature of their data density to gradually reduce the noise level.

## 2. Core Diffusion Models (2020–2021)

- **2020 NeurIPS: “Denoising Diffusion Probabilistic Models” (Ho et al.)**

  > [Paper](https://arxiv.org/abs/2006.11239) & [Website](https://hojonathanho.github.io/diffusion/) & [Video](https://slideslive.com/38936172) & [Code(official Tensorflow version)](https://github.com/hojonathanho/diffusion) & [Code(Pytorch version)](https://github.com/lucidrains/denoising-diffusion-pytorch) & [An In-Depth Guide Blog](https://learnopencv.com/denoising-diffusion-probabilistic-models/)

  ![Figure 4. The directed graphical model considered in DDPM.](./assets/figure4.png)
  
  The authors theoretically proves that the variational inference objective (maximizing log‑likelihood) and the mean‑squared‑error training for noise prediction are mathematically equivalent, unifying the two viewpoints. From the variational inference perspective, DDPM decomposes the log‑likelihood lower bound (VLB) into a series of KL divergences. From the score‑matching perspective, the network implicitly performs score matching on each noisy data distribution by predicting the added noise. Optimizing the VLB naturally yields a weighted noise‑prediction (score‑matching) objective, and conversely, directly training to predict noise also maximizes the data log‑likelihood.

  ![Figure 5. From variational inference to denoising score matching.](./assets/figure5.png)

  The training loss evolves from variational inference to denoising score matching, and Langevin dynamics is the natural sampler for denoising score matching.  
  
  ![Figure 6. Algorithms in DDPM.](./assets/figure6.png)

- **2020 ICLR: “Denoising Diffusion Implicit Models” (Song et al.)**

  > [Paper](https://arxiv.org/abs/2010.02502) & [OpenReview](https://openreview.net/forum?id=St1giarCHLP) & [Video](https://slideslive.com/38953675) & [Code](https://github.com/ermongroup/ddim)

  The authors devise a family of “skip-step” noise-injection processes, parameterized by σ, that inject the same amount of noise at each key timestep as a DDPM but no longer require strictly Markovian, one-step-at-a-time progression through every intermediate step.  

  ![Figure 7. Graphical models for diffusion(left) and non‑Markovian(right) inference models.](./assets/figure7.png)

  They further show that no matter which σ-parameterized “skip-step” noise-injection process you choose, the variational lower‑bound objective you minimize during training is exactly the same $L_1$ surrogate loss used in DDPMs. In other words, you only need to train the model once with the standard procedure, and you can then freely switch between different noise‑injection/denoising trajectories at sampling time without retraining.  

  ![Figure 8. Graphical model for accelerated generation, where $\tau=[1, 3]$.](./assets/figure8.png)

  This allows you to perform both the forward noise injection and reverse denoising only on a chosen subsequence ${\tau_1,\dots,\tau_S}$, so that when $S\ll T$, you achieve a 10×–100× speedup while maintaining high sample quality, with no retraining required.

- **2021 ICLR: “Score-Based Generative Modeling through SDEs” (Song et al.)**

  > [Paper](https://arxiv.org/pdf/2011.13456) & [OpenReview](https://openreview.net/forum?id=PxTIG12RRHS) & [Video](https://www.youtube.com/watch?v=wMmqCMwuM2Q) & [Code](https://github.com/yang-song/score_sde)

  
