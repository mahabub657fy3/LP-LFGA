Data-Efficient Multi-Target Generative Attack with Learnable Prompts

Official implementation of “Data-Efficient Multi-Target Generative Attack with Learnable Prompts” (ICME 2025 submission). 

1229-Data-Efficient Multi-Targe…

This project proposes a single conditional generator for scalable multi-target transferable attacks, integrating:

a low-pass frequency branch to bias perturbations toward transferable low-frequency content, 

1229-Data-Efficient Multi-Targe…

CLIP-guided class conditioning (precomputed prompt ensemble or learnable prompts), 

1229-Data-Efficient Multi-Targe…

and a CoOp-style prompt learner optimized for the attack objective under limited data/class coverage. 

1229-Data-Efficient Multi-Targe…

The method enforces an ℓ∞ budget and constructs adversarial examples around a low-frequency anchor with a multi-view augmentation objective. 

1229-Data-Efficient Multi-Targe…

Highlights

Data-efficient multi-target training: designed for small ImageNet subsets (e.g., 10k–100k) and subset-based class protocols (e.g., N8 on ImageNet, C5 on CIFAR-10). 

1229-Data-Efficient Multi-Targe…

Frequency-aware perturbation injection: adversarial content is centered on the low-frequency component, controlled by Gaussian low-pass parameter K. 

1229-Data-Efficient Multi-Targe…

CLIP conditioning:

Precomputed multi-prompt text features (fast, plug-and-play)

Learnable prompts (better data efficiency and transfer) 

1229-Data-Efficient Multi-Targe…

Repository Contents
.
├── train_imagenet.py
├── eval_imagenet.py
├── train_cifar10.py
├── eval_cifar10.py
├── inference.py
├── utils.py
├── prompt_learner.py
├── get_multi_prompt_text feature.py
├── imagenet_class_index.json
├── cifar10_class_index.json
├── imagenet_text_feature_multi_prompt.pth
└── cifar10_text_feature_multi_prompt.pth


Outputs

Training checkpoints (default): checkpoints_imagenet/ or checkpoints_cifar10/

Generated adversarial images (default): results_imagenet/ or results_cifar10/

expected structure:

results_imagenet/gan_n8/res50_t150/images/*.png
results_imagenet/gan_n8/res50_t426/images/*.png
...

Method Overview (Paper ↔ Code)

The training loop follows Algorithm 1 in the paper. 

1229-Data-Efficient Multi-Targe…


Key design/parameters used in experiments:

ℓ∞ budget: ε = 16/255 

1229-Data-Efficient Multi-Targe…

Gaussian low-pass kernel: K = 4 (ImageNet), K = 2 (CIFAR-10) 

1229-Data-Efficient Multi-Targe…

Conditioning: CLIP ViT-B/16 text features 

1229-Data-Efficient Multi-Targe…

Protocols: N8 (ImageNet), C5 (CIFAR-10) 

1229-Data-Efficient Multi-Targe…

Installation
1) Create environment

Recommended: Python 3.9–3.11, CUDA-enabled PyTorch.

conda create -n dep-mtga python=3.10 -y
conda activate dep-mtga

2) Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm tqdm pandas pillow numpy
pip install git+https://github.com/openai/CLIP.git


Notes

Some victim-model weights may download automatically via torchvision / torch.hub on first run.

If you are on a restricted network, pre-download weights or cache them.

Data Preparation
ImageNet (ImageFolder format)

Your scripts expect:

IMAGENET_TRAIN/
  n01440764/
    *.JPEG
  n01443537/
    *.JPEG
  ...


Set --train_dir / --data_dir accordingly.

CIFAR-10 (ImageFolder format)

Your CIFAR scripts use torchvision.datasets.ImageFolder, so prepare:

cifar10/train/<class_name>/*.png
cifar10/test/<class_name>/*.png


If you currently have CIFAR-10 in the standard binary format, convert it into ImageFolder structure first (many public converters exist).

CLIP Text Features (Precomputed)

This repo already includes:

imagenet_text_feature_multi_prompt.pth

cifar10_text_feature_multi_prompt.pth

(Optional) Regenerate text features
python "get_multi_prompt_text feature.py" \
  --dataset imagenet \
  --label_flag N8 \
  --clip_backbone ViT-B/16 \
  --save_path imagenet_text_feature_multi_prompt.pth


For CIFAR-10:

python "get_multi_prompt_text feature.py" \
  --dataset cifar10 \
  --label_flag C5 \
  --clip_backbone ViT-B/16 \
  --save_path cifar10_text_feature_multi_prompt.pth

Pretrained Weights

Place pretrained generator checkpoints under a consistent path, e.g.:

pretrained/
  imagenet/
    res50/
      model-9.pth
      prompt-9.pth          # only if using learnable prompts
  cifar10/
    resnet56/
      model-9.pth
      prompt-9.pth          # only if using learnable prompts


You can also directly pass the checkpoint paths via --load_path and --prompt_ckpt.

One-Click Evaluation (Pretrained → Generate → Inference)

Evaluation is a two-step pipeline:

Generate adversarial images (eval_*.py)

Compute TSR over victim models (inference.py)

A) ImageNet (N8)
Step 1 — Generate adversarial images

Precomputed prompts (recommended for quick eval):

python eval_imagenet.py \
  --data_dir /path/to/imagenet/val \
  --model_type res50 \
  --label_flag N8 \
  --eps 16 \
  --batch_size 10 \
  --k 4 \
  --prompt_mode precomputed \
  --load_path pretrained/imagenet/res50/model-9.pth \
  --save_dir results_imagenet


Learnable prompts:

python eval_imagenet.py \
  --data_dir /path/to/imagenet/val \
  --model_type res50 \
  --label_flag N8 \
  --eps 16 \
  --batch_size 10 \
  --k 4 \
  --prompt_mode learnable \
  --prompt_ckpt pretrained/imagenet/res50/prompt-9.pth \
  --load_path pretrained/imagenet/res50/model-9.pth \
  --save_dir results_imagenet

Step 2 — Run inference (TSR)
python inference.py \
  --dataset imagenet \
  --label_flag N8 \
  --model_t normal \
  --batch_size 10 \
  --test_dir results_imagenet/gan_n8/res50


--model_t options:

normal: vgg16, googlenet, incv3, res50, res152, dense121

robust: robust variants

all: includes both (plus more)

cifar: CIFAR-only list

B) CIFAR-10 (C5)
Step 1 — Generate adversarial images
python eval_cifar10.py \
  --data_dir /path/to/cifar10/test \
  --model_type cifar10_resnet56 \
  --label_flag C5 \
  --eps 16 \
  --batch_size 10 \
  --k 2 \
  --prompt_mode precomputed \
  --load_path pretrained/cifar10/resnet56/model-9.pth \
  --save_dir results_cifar10

Step 2 — Run inference (TSR)
python inference.py \
  --dataset cifar10 \
  --label_flag C5 \
  --model_t cifar \
  --batch_size 10 \
  --test_dir results_cifar10/gan_n8/cifar10_resnet56

Training

Training saves:

Generator: model-{epoch}.pth

Prompt learner (if enabled): prompt-{epoch}.pth

A) Train on ImageNet
python train_imagenet.py \
  --train_dir /path/to/imagenet/train \
  --model_type res50 \
  --label_flag N8 \
  --epochs 10 \
  --batch_size 8 \
  --lr 2e-4 \
  --eps 16 \
  --k 4 \
  --prompt_mode learnable \
  --save_dir checkpoints_imagenet


To resume:

python train_imagenet.py \
  --train_dir /path/to/imagenet/train \
  --model_type res50 \
  --label_flag N8 \
  --start_epoch 5 \
  --load_path checkpoints_imagenet/res50_N8/model-4.pth \
  --prompt_ckpt checkpoints_imagenet/res50_N8/prompt-4.pth \
  --save_dir checkpoints_imagenet

B) Train on CIFAR-10
python train_cifar10.py \
  --train_dir /path/to/cifar10/train \
  --model_type cifar10_resnet56 \
  --label_flag C5 \
  --epochs 10 \
  --batch_size 8 \
  --lr 2e-4 \
  --eps 16 \
  --k 2 \
  --prompt_mode learnable \
  --save_dir checkpoints_cifar10

Reproducibility Notes

Default attack budget is ε = 16/255. 

1229-Data-Efficient Multi-Targe…

Default low-pass kernel uses K = 4 (ImageNet) and K = 2 (CIFAR-10), matching the paper. 

1229-Data-Efficient Multi-Targe…

Protocols (N8, C5) are used for multi-target evaluation averages. 
