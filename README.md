# LP-LFGA: Data-Efficient Multi-Target Generative Attack with Learnable Prompts

<p align="center">
  <img src="![diagram](https://github.com/user-attachments/assets/61cc936e-b904-4d4a-afad-ec5ec1e97a14)
" alt="LP-LFGA overall framework" width="920">
</p>

<p align="center">
  <b>LP-LFGA</b> is a data-efficient, multi-target, conditional generative attack framework that integrates <b>frequency decomposition</b> and <b>CLIP-guided conditioning</b> with <b>learnable prompts</b>.
</p>

---

## Abstract (from the paper)

Deep Neural Networks (DNNs) have achieved 
remarkable success in vision applications, yet remain highly 
vulnerable to adversarial examples, posing serious risks for safety -
critical systems such as autonomous driving and biometric 
authentication. Transfer -based attacks are particularly 
concerning because an adversary can craft adversarial examples 
on a surrogate model and reliably fool unseen black -box models 
without querying them. However, existing transferable targeted 
attacks either require trai ning one generator per target class 
which is computationally prohibitive at scale , or ignore rich 
semantic priors thus suffer from limited transferability. In this 
paper, we propose a data-efficient multi -target generative attack 
with learnable prompts , which integrates frequency decomposition 
and CLIP -guided conditioning. Technically, we design (i) a low -
pass frequency branch that operates on the smoothed image to 
reduce overfitting to high -frequency noise, (ii) a CLIP -based 
conditional generator that injects class -dependent text features at 
multiple feature levels, and (iii) a CoOp -style prompt learner that 
adapts CLIP text embeddings to the attack objective using only a 
small subset of classes and images. On ImageNet and CIFAR -10, 
our method achieves cons istently higher targeted transfer success 
rates than state -of-the-art multi -target generative attacks , while 
requiring only a single conditional generator. We further show 
that learnable prompts improve data efficiency under limited 
training data and scarc e class coverage, and that our frequency -
aware generator yields stronger robustness to input 
transformations and robust -training defenses. The code is 
available at: xxxxx  
Keywords —component, formatting, style, styling, insert ( key 
words )

---

## What’s inside

- End-to-end pipeline:
  - **Train** a single conditional generator (multi-target)
  - **Evaluate** by generating adversarial images
  - **Inference** to compute targeted transfer success rate (TSR)
- Conditioning options:
  - **Learnable prompts** 
  - **Precomputed multi-prompt CLIP text features**
- Datasets supported by scripts:
  - **ImageNet**
  - **CIFAR-10**

---

> Tip: Add `data/`, `results/`, `checkpoints/` to `.gitignore`.

---

## Environment setup (one command)

This project ships a pinned Conda environment: **`environment.yml`**.
Python: 3.10
PyTorch: 2.2.1 + CUDA 11.8
torchvision: 0.17.1
torchaudio: 2.2.1
CLIP: 1.0
einops: 0.8.1
### 1) Create and activate
```bash
conda env create -f environment.yml
conda activate LP-LFGA
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda:', torch.version.cuda)"
Environment summary (from environment.yml)

mkdir -p pretrained/imagenet pretrained/cifar10

# ImageNet (N8, ResNet-50 surrogate)
# wget -O pretrained/imagenet/res50_N8_eps16_k4.zip  <YOUR_RELEASE_URL>
# unzip pretrained/imagenet/res50_N8_eps16_k4.zip -d pretrained/imagenet/

# CIFAR-10 (C5, ResNet-56 surrogate)
# wget -O pretrained/cifar10/resnet56_C5_eps16_k2.zip <YOUR_RELEASE_URL>
# unzip pretrained/cifar10/resnet56_C5_eps16_k2.zip -d pretrained/cifar10/
Quick start: one-click evaluation (pretrained → generate → inference)

Evaluation is a 2-stage pipeline:

eval_*.py generates adversarial images

inference.py computes TSR on a model set

You can run both with && as a one-click command.

A) ImageNet (recommended: learnable prompts)
python eval_imagenet.py \
  --data_dir /path/to/imagenet/val \
  --label_flag N8 \
  --model_type res50 \
  --eps 16 \
  --k 4 \
  --batch_size 10 \
  --prompt_mode learnable \
  --load_path pretrained/imagenet/res50_N8_eps16_k4/model.pth \
  --prompt_ckpt pretrained/imagenet/res50_N8_eps16_k4/prompt.pth \
  --save_dir results/imagenet \
&& \
python inference.py \
  --dataset imagenet \
  --label_flag N8 \
  --model_t normal \
  --batch_size 10 \
  --test_dir results/imagenet

B) CIFAR-10 (recommended: learnable prompts)
python eval_cifar10.py \
  --data_dir /path/to/cifar10/test \
  --label_flag C5 \
  --model_type cifar10_resnet56 \
  --eps 16 \
  --k 2 \
  --batch_size 10 \
  --prompt_mode learnable \
  --load_path pretrained/cifar10/resnet56_C5_eps16_k2/model.pth \
  --prompt_ckpt pretrained/cifar10/resnet56_C5_eps16_k2/prompt.pth \
  --save_dir results/cifar10 \
&& \
python inference.py \
  --dataset cifar10 \
  --label_flag C5 \
  --model_t cifar \
  --batch_size 10 \
  --test_dir results/cifar10


Tip: If your script argument names differ slightly, run:
python eval_imagenet.py -h and python inference.py -h.

Training

Training saves:

Generator weights: model-<epoch>.pth

Prompt weights (learnable prompts): prompt-<epoch>.pth

Train on ImageNet
python train_imagenet.py \
  --train_dir /path/to/imagenet/train \
  --label_flag N8 \
  --model_type res50 \
  --epochs 10 \
  --batch_size 8 \
  --lr 2e-4 \
  --eps 16 \
  --k 4 \
  --prompt_mode learnable \
  --save_dir checkpoints/imagenet

Train on CIFAR-10
python train_cifar10.py \
  --train_dir /path/to/cifar10/train \
  --label_flag C5 \
  --model_type cifar10_resnet56 \
  --epochs 10 \
  --batch_size 128 \
  --lr 2e-4 \
  --eps 16 \
  --k 2 \
  --prompt_mode learnable \
  --save_dir checkpoints/cifar10

Inference / metrics

inference.py reports targeted transfer success rates (TSR) across selected target model groups via --model_t.

Common options:

--model_t normal : standard models

--model_t robust : robust models

--model_t all : all available target models

--model_t cifar : CIFAR-only targets

Example:

python inference.py \
  --dataset imagenet \
  --label_flag N8 \
  --model_t all \
  --batch_size 10 \
  --test_dir results/imagenet

Figures

Overall framework (recommended path): assets/framework.png

Add additional paper figures (optional):

assets/teaser.png

assets/qualitative.png

If you want, you can also add a “Results” section below and embed qualitative examples:

## Qualitative Results
<p align="center">
  <img src="assets/qualitative.png" width="920">
</p>

Notes / troubleshooting
(1) Missing modules / import errors

If you encounter ModuleNotFoundError, ensure your repo contains the expected modules referenced in the scripts
(or update import paths accordingly). Typical examples:

models/generator.py (Generator)

image_transformer.py (rotation)

(2) Precomputed prompt mode

If you support a --prompt_mode precomputed, ensure the code exposes a --text_feature_path argument and loads
the correct .pth file (e.g., imagenet_text_feature_multi_prompt.pth).
