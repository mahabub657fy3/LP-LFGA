Data-Efficient Multi-Target Generative Attack with Learnable Prompts

Official implementation of "Data-Efficient Multi-Target Generative Attack with Learnable Prompts" - A novel adversarial attack framework that integrates frequency decomposition and CLIP-guided conditioning for highly transferable targeted attacks.

📖 Abstract

Deep Neural Networks (DNNs) have achieved remarkable success in vision applications, yet remain highly vulnerable to adversarial examples, posing serious risks for safety-critical systems such as autonomous driving and biometric authentication. Transfer-based attacks are particularly concerning because an adversary can craft adversarial examples on a surrogate model and reliably fool unseen black-box models without querying them. However, existing transferable targeted attacks either require training one generator per target class which is computationally prohibitive at scale, or ignore rich semantic priors thus suffer from limited transferability. 

In this paper, we propose a data-efficient multi-target generative attack with learnable prompts, which integrates frequency decomposition and CLIP-guided conditioning. Technically, we design:
• Low-pass frequency branch that operates on the smoothed image to reduce overfitting to high-frequency noise

• CLIP-based conditional generator that injects class-dependent text features at multiple feature levels

• CoOp-style prompt learner that adapts CLIP text embeddings to the attack objective using only a small subset of classes and images

On ImageNet and CIFAR-10, our method achieves consistently higher targeted transfer success rates than state-of-the-art multi-target generative attacks, while requiring only a single conditional generator. We further show that learnable prompts improve data efficiency under limited training data and scarce class coverage, and that our frequency-aware generator yields stronger robustness to input transformations and robust-training defenses.

🏗️ Overall Framework

<div align="center">
  <img width="1067" height="273" alt="diagram" src="https://github.com/user-attachments/assets/2b7e6ad7-682b-4a13-807a-f470643c7329" />
</div>

Our proposed framework consists of three key components: (1) Low-pass frequency decomposition to extract robust features, (2) CLIP-based conditional generator with multi-level feature injection, and (3) Learnable prompt module that adapts text embeddings for attack optimization.

🚀 Quick Start

Prerequisites

• Python 3.10+

• PyTorch 2.2.1+

• CUDA 11.8+

• 8GB+ GPU memory

Installation

Environment Setup
1. Install Dependencies
   First, set up the environment using the provided environment.yml file:
   conda env create -f environment.yml
   conda activate lp-lfga
   
3. Download Pretrained Models
You can download the pretrained models directly from the link provided below or upload your own model .pth file.
Download Pretrained Models

Or upload your own .pth files under the models/ directory.
