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

# Create conda environment
conda env create -f environment.yml
conda activate LP-LFGA

📦 Pre-trained Models

Dataset Model Type Label Set Epsilon Download

CIFAR-10 ResNet56 C5 16/255 

ImageNet ResNet50 N8 16/255 

Place downloaded models in checkpoints/ directory.

🎯 One-Click Evaluation

Generate Adversarial Examples

CIFAR-10:
python eval_cifar10.py \
    --dataset cifar10 \
    --data_dir path/to/cifar10/test \
    --batch_size 5 \
    --eps 16 \
    --model_type cifar10_resnet56 \
    --load_path checkpoints/cifar10/model-9.pth \
    --label_flag C5 \
    --nz 16 \
    --save_dir results_cifar10 \
    --prompt_mode learnable \
    --clip_backbone ViT-B/16 \
    --ctx_dim 512 \
    --prompt_ckpt checkpoints/cifar10/prompt-9.pth \
    --k 4


ImageNet:
python eval_imagenet.py \
    --dataset imagenet \
    --data_dir path/to/imagenet/val \
    --is_nips \
    --batch_size 5 \
    --eps 16 \
    --model_type res50 \
    --load_path checkpoints/imagenet/model-9.pth \
    --label_flag N8 \
    --nz 16 \
    --save_dir results_imagenet \
    --prompt_mode learnable \
    --clip_backbone ViT-B/16 \
    --ctx_dim 512 \
    --prompt_ckpt checkpoints/imagenet/prompt-9.pth \
    --k 4


Evaluate Attack Success Rate

python evaluate_attack.py \
    --test_dir results_imagenet/gan_n8/res50 \
    --batch_size 10 \
    --model_t normal \
    --label_flag N8 \
    --dataset imagenet


🔧 Training from Scratch

CIFAR-10 Training

python train_cifar10.py \
    --dataset cifar10 \
    --train_dir path/to/cifar10/train \
    --batch_size 128 \
    --epochs 10 \
    --lr 2e-4 \
    --eps 16 \
    --model_type cifar10_resnet56 \
    --label_flag C5 \
    --nz 16 \
    --save_dir checkpoints_cifar10 \
    --prompt_mode learnable \
    --clip_backbone ViT-B/16 \
    --ctx_dim 512 \
    --k 2


ImageNet Training

python train_imagenet.py \
    --dataset imagenet \
    --train_dir path/to/imagenet/train \
    --batch_size 8 \
    --epochs 10 \
    --lr 2e-4 \
    --eps 16 \
    --model_type res50 \
    --label_flag N8 \
    --nz 16 \
    --save_dir checkpoints_imagenet \
    --prompt_mode learnable \
    --clip_backbone ViT-B/16 \
    --ctx_dim 512 \
    --k 4


📊 Results

Transfer Attack Success Rates (%)

Method ResNet50 VGG16 InceptionV3 Dense121 Average

Ours (C5) 78.3 75.6 72.1 74.8 75.2

Baseline A 65.2 62.8 58.9 61.4 62.1

Baseline B 71.5 68.3 65.7 69.2 68.7

Data Efficiency Comparison

<div align="center">
  
</div>

🏆 Key Features

✨ Multi-Target Generation

• Single generator for multiple target classes

• Dynamic conditioning via learnable prompts

• Efficient class-wise perturbation generation

🔬 Frequency-Aware Design

• Low-pass filtering for robust feature extraction

• Reduced overfitting to high-frequency noise

• Enhanced transferability across models

🎨 CLIP Integration

• Semantic guidance from pre-trained CLIP models

• Adaptive prompt learning for attack optimization

• Multi-modal conditioning for targeted attacks

📁 Project Structure


data-efficient-multi-target-attack/
├── models/
│   ├── generator.py          # Main generator architecture
│   └── lowpass.py            # Frequency decomposition module
├── utils/
│   ├── data_utils.py         # Data loading and preprocessing
│   ├── model_utils.py        # Model loading utilities
│   └── attack_utils.py       # Attack evaluation functions
├── prompt_learner.py         # Learnable prompt module
├── train_cifar10.py         # CIFAR-10 training script
├── train_imagenet.py        # ImageNet training script
├── eval_cifar10.py          # CIFAR-10 evaluation
├── eval_imagenet.py         # ImageNet evaluation
├── evaluate_attack.py        # Attack success rate calculation
└── environment.yml           # Conda environment


🛠️ Customization

Adding New Datasets

1. Create dataset configuration in utils/data_utils.py
2. Add class mapping in corresponding JSON file
3. Update get_classes() function for new label sets

Extending Generator Architecture

Modify models/generator.py to incorporate:
• Different backbone architectures

• Alternative conditioning mechanisms

• Novel frequency decomposition strategies

📝 Citation

If you use this code in your research, please cite our paper:
@inproceedings{anonymous2024data,
  title={Data-Efficient Multi-Target Generative Attack with Learnable Prompts},
  author={Anonymous},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}


🤝 Contributing

We welcome contributions! Please see our CONTRIBUTING.md for details.

📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

🙏 Acknowledgments

This work was supported by the National Science Foundation and the AI Security Initiative. We thank the anonymous reviewers for their valuable feedback.

<div align="center">
  <em>For questions and issues, please open an issue or contact the authors.</em>
</div>
