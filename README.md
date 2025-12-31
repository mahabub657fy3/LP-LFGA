---
# Data-Efficient Multi-Target Generative Attack with Learnable Prompts**

## **Abstract**

Deep Neural Networks (DNNs) are critically vulnerable to adversarial examples, posing significant risks in safety-critical applications such as autonomous driving and biometric authentication. Recent advancements in generative adversarial attacks highlight their effectiveness, particularly due to their transferability and efficient inference. However, most of the existing methods assume the availability of a large portion of the victim model's training data, which is an unrealistic scenario in real-world applications. To address this limitation, we propose a data-efficient approach that combines frequency decomposition and CLIP-guided conditioning for adversarial attacks. Our framework incorporates (i) a low-pass frequency branch to perturb a neutralized image and facilitate optimization, (ii) a conditional generator that leverages CLIP text embeddings for target-class semantics, and (iii) a learnable prompt module to adapt these embeddings to the attack objective with only a small subset of classes and images. Experimental results on ImageNet and CIFAR-10 datasets demonstrate that our method achieves state-of-the-art targeted transferability, outperforming existing multi-target attacks, particularly in data-scarce scenarios.

---

## **Features**

* **One-click Evaluation:** Easily evaluate the performance of the adversarial attack on CIFAR-10 or ImageNet datasets.
* **One-click Inference:** Generate adversarial examples for specified classes using a pre-trained or trained model.
* **One-click Training:** Train the model from scratch or resume from a checkpoint with minimal setup.
* **Pretrained Models:** Download pretrained models from Google Drive or upload your own model checkpoint (`.pth` file).

---

## **Pipeline Overview**

The overall pipeline for the LP-LFGA framework is as follows:

1. **Data Input:** Input images are processed through data augmentation and normalization.
2. **Frequency Decomposition:** The low-frequency components of the image are extracted using a Gaussian filter.
3. **CLIP-Guided Conditioning:** The model is conditioned using text embeddings from CLIP, either precomputed or learned through a prompt module.
4. **Adversarial Generation:** The conditional generator produces adversarial examples that perturb the low-frequency components.
5. **Model Evaluation:** The generated adversarial examples are evaluated for transferability and effectiveness against the victim model.

*<img width="1064" height="260" alt="diagram" src="https://github.com/user-attachments/assets/e2302b0f-fc51-4795-a43c-8aef21e40428" />*

---

## **Installation**

### **1. Setup Environment**

First, clone the repository and set up the required environment using the provided `environment.yml` file:

```bash
git clone https://github.com/mahabub657fy3/LP-LFGA
cd LP-LFGA
conda env create -f environment.yml
conda activate lp-lfga
```

### **2. Download Pretrained Models**

You can download the pretrained models directly from Google Drive or upload your own `.pth` files.

* **[Download Pretrained Models] **

Or, upload your own `.pth` files to the `models/` directory.

---

## **Usage**

### **One-click Evaluation**

Generate Adversarial Examples on CIFAR-10 or ImageNet, use the following commands:

#### **CIFAR-10 Evaluation:**
Below we provide running commands for generating targeted adversarial examples on [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) (10k images)  under our multi-class setting:

```bash
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
    --k 2
```

#### **ImageNet Evaluation:**
Below we provide running commands for generating targeted adversarial examples on [ImageNet NeurIPS validation set](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack) (1k images) under our multi-class setting:

```bash
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
```

This will load the pretrained model, Generate adversarial examples using the trained generator, and store the generated samples in the `results/` directory.

---

### **One-click Inference**

 Here are the commands to run inference to Evaluate Target Success Rate:

#### **CIFAR-10 Inference:**

```bash
python inference.py --dataset cifar10 --data_dir ./cifar10/test --batch_size 5 --eps 16 --model_type cifar
```

#### **ImageNet Inference:**

```bash
python inference.py --dataset imagenet --data_dir ./imagenet/test --batch_size 5 --eps 16 --model_t normal --save_dir ./output_imagenet
```

This will Evaluate Attack Success Rate.

---

### **One-click Training**

To train the model from scratch or resume from a checkpoint, use the following commands:

#### **CIFAR-10 Training:**

```bash
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
```

#### **ImageNet Training:**

```bash
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
```

The model will be trained for the specified number of epochs and saved periodically in the `checkpoints/` directory.

---
