# Latent_CFM
This repository contains the codebase of the paper "Efficient Flow Matching using Latent Variables" ([https://arxiv.org/abs/2505.04486](https://arxiv.org/abs/2505.04486)).

**Abstarct**
Flow matching models have shown great potential in image generation tasks among probabilistic generative models. However, most flow matching models in the literature do not explicitly model the underlying structure/manifold in the target data when learning the flow from a simple source distribution like the standard Gaussian. This leads to inefficient learning, especially for many high-dimensional real-world datasets, which often reside in a low-dimensional manifold. Existing strategies of incorporating manifolds, including data with underlying multi-modal distribution, often require expensive training and hence frequently lead to suboptimal performance. To this end, we present 𝙻𝚊𝚝𝚎𝚗𝚝-𝙲𝙵𝙼, which provides simplified training/inference strategies to incorporate multi-modal data structures using pretrained deep latent variable models. Through experiments on multi-modal synthetic data and widely used image benchmark datasets, we show that 𝙻𝚊𝚝𝚎𝚗𝚝-𝙲𝙵𝙼 exhibits improved generation quality with significantly less training (up to ∼50% less) and computation than state-of-the-art flow matching models by incorporating extracted data features using pretrained lightweight latent variable models. Moving beyond natural images to generating fields arising from processes governed by physics, using a 2d Darcy flow dataset, we demonstrate that our approach generates more physically accurate samples than competitive approaches. In addition, through latent space analysis, we demonstrate that our approach can be used for conditional image generation conditioned on latent features, which adds interpretability to the generation process.

<div align="center">
  <img src="https://github.com/AnirbanSamaddar/Latent_CFM/blob/main/img/Schematic4.png?raw=true" width="700" height="500" />
</div>

**Figure:** Schematic of Latent-CFM framework. Given a data x1, 𝙻𝚊𝚝𝚎𝚗𝚝-𝙲𝙵𝙼 extracts latent features using a frozen encoder and a trainable stochastic layer. The features are embedded using
a linear layer and added to the learned vector field. The framework resembles an encoder-decoder architecture like VAEs.

