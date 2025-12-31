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

I can help you create a comprehensive README file for your GitHub repository. The README will include sections for one-click evaluation, one-click inference, one-click training, and instructions for downloading pretrained models.

Here's a draft of the README:

---

# **LP-LFGA: Data-Efficient Multi-Target Generative Attack with Learnable Prompts**

## **Overview**

This repository implements a data-efficient multi-target generative attack using learnable prompts. The framework combines frequency decomposition with CLIP-guided conditioning to achieve state-of-the-art performance in targeted transferability under limited training data. It performs adversarial attacks on both ImageNet and CIFAR-10 datasets using pre-trained models or models trained on a small subset of data.

## **Features**

* **One-click Evaluation:** Evaluate the performance of the adversarial attack on the specified dataset (CIFAR-10, ImageNet).
* **One-click Inference:** Generate adversarial examples for specified classes using the trained generator.
* **One-click Training:** Train the model from scratch or from a checkpoint.
* **Pretrained Models:** Download pretrained models from Google Drive or upload your own `.pth` file.

## **Environment Setup**

### **1. Install Dependencies**

First, set up the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate lp-lfga
```

Alternatively, you can manually install the necessary dependencies:

```bash
pip install torch torchvision numpy tqdm matplotlib
pip install git+https://github.com/cvg/LightGlue.git
```

### **2. Download Pretrained Models**

You can download the pretrained models directly from the link provided below or upload your own model `.pth` file.

* **[Download Pretrained Models](https://drive.google.com/drive/folder_link)**

Or upload your own `.pth` files under the `models/` directory.

---

## **One-click Evaluation**

To evaluate the performance of the model on CIFAR-10 or ImageNet, use the provided evaluation scripts.

### **CIFAR-10 Evaluation:**

```bash
python eval_cifar10.py --dataset cifar10 --data_dir ./cifar10/test --batch_size 5 --eps 16 --model_type cifar10_resnet56 --load_path ./models/model-9.pth
```

### **ImageNet Evaluation:**

```bash
python eval_imagenet.py --dataset imagenet --data_dir ./imagenet/test --batch_size 5 --eps 16 --model_type res50 --load_path ./models/model-9.pth
```

This will load the pretrained model, evaluate the performance, and save the results to the `results/` directory.

---

## **One-click Inference**

You can generate adversarial examples for specified target classes using the trained generator. Here’s how to run inference:

### **CIFAR-10 Inference:**

```bash
python inference.py --dataset cifar10 --data_dir ./cifar10/test --batch_size 5 --eps 16 --model_type cifar10_resnet56 --load_path ./models/model-9.pth --save_dir ./output_cifar10
```

### **ImageNet Inference:**

```bash
python inference.py --dataset imagenet --data_dir ./imagenet/test --batch_size 5 --eps 16 --model_type res50 --load_path ./models/model-9.pth --save_dir ./output_imagenet
```

This will generate adversarial images for the specified target classes and save them to the `output/` directory.

---

## **One-click Training**

To train the model from scratch or resume from a checkpoint, run the following command:

### **CIFAR-10 Training:**

```bash
python train_cifar10.py --dataset cifar10 --train_dir ./cifar10/train --batch_size 128 --epochs 10 --lr 2e-4 --eps 16 --model_type cifar10_resnet56 --save_dir ./checkpoints_cifar10
```

### **ImageNet Training:**

```bash
python train_imagenet.py --dataset imagenet --train_dir ./imagenet/train --batch_size 8 --epochs 10 --lr 2e-4 --eps 16 --model_type res50 --save_dir ./checkpoints_imagenet
```

The model will be trained for the specified number of epochs and saved periodically to the `checkpoints/` directory.

---

## **Directory Structure**

```bash
LP-LFGA/
├── checkpoints/                # Trained models
├── models/                     # Pretrained model files (.pth)
├── output/                     # Generated adversarial examples
├── results/                    # Evaluation results
├── eval_cifar10.py             # CIFAR-10 evaluation script
├── eval_imagenet.py            # ImageNet evaluation script
├── inference.py                # Inference script for adversarial examples
├── train_cifar10.py            # Training script for CIFAR-10
├── train_imagenet.py           # Training script for ImageNet
└── environment.yml             # Environment setup file
```

---

## **Usage Instructions**

1. **Download or Upload Pretrained Models:** You can download pretrained models or upload your own `.pth` files to the `models/` directory.
2. **Run One-click Evaluation:** Use `eval_cifar10.py` or `eval_imagenet.py` to evaluate the model on CIFAR-10 or ImageNet.
3. **Run One-click Inference:** Use `inference.py` to generate adversarial examples.
4. **Train the Model:** Use `train_cifar10.py` or `train_imagenet.py` to train the model from scratch.

---

## **Acknowledgements**

* The code is built upon previous works such as LightGlue and various adversarial attack papers. For more details, please refer to the `Data-Efficient Multi-Target Generative Attack with Learnable Prompts` paper.

---

This should be a comprehensive and well-organized README file for your repository. Would you like me to make any additional changes?

