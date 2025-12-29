import argparse
import json
import torch
import clip
from tqdm import tqdm
from utils import get_classes

# CIFAR-10 class names in label order (0 ~ 9)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

def load_imagenet_classnames(json_path):
    with open(json_path, "r") as f:
        idx_to_info = json.load(f)

    # idx: int  ->  class_name: str
    idx_to_name = {}
    for idx_str, (wnid, class_name) in idx_to_info.items():
        idx = int(idx_str)
        class_name_clean = class_name.split(",")[0]
        idx_to_name[idx] = class_name_clean

    return idx_to_name


IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

# CIFAR-10 templates from CLIP prompts.md
CIFAR10_TEMPLATES = [
        "a photo of a {}",
        "a blurry photo of a {}",
        "a black and white photo of a {}",
        "a close-up photo of a {}",
        "a cropped photo of a {}",
        "a photo of a small {}",
        "a photo of a big {}",
        "a bright photo of a {}",
        "a dark photo of a {}",
        "a painting of a {}",
        "an artwork of a {}",
        "a sculpture of a {}",
        "a cartoon of a {}",
        "a rendering of a {}",
        "a clean photograph of a {}",
        "a dirty photograph of a {}",
        "a photo of the {}",
        "a photo of my {}",
        "a good photo of a {}",
        "a close-up photo of the {}",
        "a drawing of a {}",
        "graffiti of a {}",
        "a toy {}",
        "a plastic {}",
        "a 3d render of a {}",
]

def build_multi_prompt_text_features(
    model_name='ViT-B/16',
    dataset='imagenet',              # 'imagenet' or 'cifar10'
    imagenet_json='imagenet_class_index.json',
    cifar10_json='cifar10_class_index.json',
    label_flag='N8',
    save_path='imagenet_text_feature_multi_prompt.pth',
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Load CLIP model
    clip_model, _ = clip.load(model_name, device=device)
    clip_model.eval()

    if dataset.lower() == 'imagenet':
        idx_to_name = load_imagenet_classnames(imagenet_json)
        templates = IMAGENET_TEMPLATES
        label_set = get_classes(label_flag, dataset=args.dataset)
    elif dataset.lower() == 'cifar10':
        idx_to_name = load_imagenet_classnames(cifar10_json)
        templates = CIFAR10_TEMPLATES
        label_set = get_classes(label_flag, dataset=args.dataset)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'imagenet' or 'cifar10'.")

    label_set = sorted(list(label_set))
    print(f"Dataset: {dataset}, label_flag: {label_flag}, num classes: {len(label_set)}")
    print(f"Using {len(templates)} templates per class.")
    print(f'Class IDs for attack: {label_set}')

    text_cond_dict = {}

    with torch.no_grad():
        for label in tqdm(label_set, desc="Building multi-prompt text features"):
            class_name = idx_to_name[label]  # e.g., "goldfish", "dog", "airplane"
            prompts = [temp.format(class_name) for temp in templates]
            tokenized = clip.tokenize(prompts).to(device)

            # Encode with CLIP
            text_embed = clip_model.encode_text(tokenized)  # [num_templates, 512]
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_mean = text_embed.mean(dim=0)
            text_embed_mean = text_embed_mean / text_embed_mean.norm()

            text_cond_dict[int(label)] = text_embed_mean.cpu()

    torch.save(text_cond_dict, save_path)
    print(f"Saved multi-prompt text features for {len(label_set)} classes to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ViT-B/16')
    parser.add_argument('--dataset', type=str, default='cifar10',choices=['imagenet', 'cifar10'])
    parser.add_argument('--imagenet_json', type=str, default='imagenet_class_index.json')
    parser.add_argument('--cifar10_json', type=str, default='cifar10_class_index.json')
    parser.add_argument('--label_flag', type=str, default='N8',help='Subset flag imagenet:N8, C20,...,C200, Subset flag imagenet:ALL, C5')
    parser.add_argument('--save_path', type=str, default='imagenet_text_feature_multi_prompt.pth')
    args = parser.parse_args()

    build_multi_prompt_text_features(
        model_name=args.model_name,
        dataset=args.dataset,
        imagenet_json=args.imagenet_json,
        cifar10_json=args.cifar10_json,
        label_flag=args.label_flag,
        save_path=args.save_path,)