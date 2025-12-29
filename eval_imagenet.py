import argparse
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from utils import *
from models.generator import Generator


def main():
    parser = argparse.ArgumentParser(description='Data-Efficient Multi-Target Generative Attack with Learnable Prompts- Evaluation')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--data_dir', type=str,default='D:/ImageNet/neurips2017_dev')
    parser.add_argument('--is_nips', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--eps', type=int, default=16)
    parser.add_argument('--model_type', type=str, default='incv3',help='res50, incv3')
    parser.add_argument('--load_path', type=str, default='model-9.pth', help='Path to generator checkpoint')
    parser.add_argument('--label_flag', type=str, default='N8', help='ImageNet: N8,C20,C50,C100,C200')
    parser.add_argument('--nz', type=int, default=16, help='Latent/cond dim for generator')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--prompt_mode', type=str, default='learnable',choices=['precomputed', 'learnable'],help='How to obtain CLIP text features')
    parser.add_argument('--clip_backbone', type=str, default='ViT-B/16')
    parser.add_argument('--ctx_dim', type=int, default=512)
    parser.add_argument('--prompt_ckpt', type=str, default='prompt-9.pth',help='Checkpoint of LearnablePrompt (for prompt_mode=learnable)')
    parser.add_argument('--k',type=int, default=4)

    args = parser.parse_args()
    print(args)

    eps = args.eps / 255.0
    k_for_gen = args.k if args.k > 0 else None

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Build generator----
    if args.model_type == 'incv3':
        scale_size, img_size = 256, 255
        netG = Generator(inception=True, nz=args.nz, k=k_for_gen, device=device)
    else:
        scale_size, img_size = 256, 224
        netG = Generator(nz=args.nz, k=k_for_gen, device=device)

    # Load Generator
    print(f"Loading generator from {args.load_path}")
    state_dict = torch.load(args.load_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG = netG.to(device)
    netG.eval()

    # ---- Data ----
    data_transform = transforms.Compose([transforms.Resize(scale_size),transforms.CenterCrop(img_size),transforms.ToTensor(),])
    test_set = datasets.ImageFolder(args.data_dir, data_transform)

    # Fix labels if needed
    if args.is_nips:
        test_set = fix_labels_nips(args, test_set, pytorch=True)
    else:
        test_set = fix_labels(args, test_set)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)

    # ---- Choose global class IDs to attack ----
    class_ids = get_classes(args.label_flag, dataset=args.dataset)
    print(f"Class IDs for attack: {class_ids}")

    # ---- Prompt / text condition setup ----
    prompt_learner = None
    global_to_local = None

    if args.prompt_mode == 'precomputed':
        text_feature_path = 'imagenet_text_feature_multi_prompt.pth'

        print(f"Loading precomputed text features from: {args.text_feature_path}")
        text_cond_dict = torch.load(args.text_feature_path,map_location=device)
        # ensure on device
        for k, v in text_cond_dict.items():
            if isinstance(v, torch.Tensor):
                text_cond_dict[k] = v.to(device)
    else:
        from prompt_learner import LearnablePrompt

        class_index = getClassIndex(args.dataset)
        used_ids = np.array(sorted(class_ids.tolist()))
        classnames = [class_index[int(i)][1] for i in used_ids]

        print("Using LearnablePrompt for classes:")
        for gid, name in zip(used_ids, classnames):
            print(f"global id {int(gid):4d} -> '{name}'")

        prompt_learner = LearnablePrompt(classnames=classnames,clip_backbone=args.clip_backbone,device=device,ctx_dim=args.ctx_dim,).to(device)

        if args.prompt_ckpt is None:
            raise ValueError("You must provide --prompt_ckpt when prompt_mode='learnable'")
        print(f"Loading LearnablePrompt checkpoint from: {args.prompt_ckpt}")
        prompt_state = torch.load(args.prompt_ckpt, map_location=device)
        prompt_learner.load_state_dict(prompt_state)
        prompt_learner.eval()

        global_to_local = {int(gid): idx for idx, gid in enumerate(used_ids)}

    # ---- Generate & save adversarial images ----
    netG.eval()
    if prompt_learner is not None:
        prompt_learner.eval()

    for idx, target_class in enumerate(class_ids):
        target_class = int(target_class)
        print(f"Generating adv examples for target class {target_class} "
              f"({idx+1}/{len(class_ids)})")

        for i, (img, _) in enumerate(test_loader):
            img = img.to(device)
            b = img.size(0)

            # Build text condition 'cond' depending on prompt mode
            if args.prompt_mode == 'precomputed':
                base_feat = text_cond_dict[target_class]  # [D]
                cond = base_feat.unsqueeze(0).expand(b, -1)  # [B, D]
            else:
                # learnable prompts
                local_idx = torch.full((b,),global_to_local[target_class],dtype=torch.long,device=device)
                cond = prompt_learner(local_idx)  # [B, D]

            adv = netG(img, cond, eps=eps)
            adv = torch.min(torch.max(adv, img - eps), img + eps)
            adv = torch.clamp(adv, 0.0, 1.0)

            save_imgs = adv.detach().cpu()
            for j in range(len(save_imgs)):
                g_img = transforms.ToPILImage('RGB')(save_imgs[j])

                output_dir = os.path.join(args.save_dir,'gan_n8', f'{args.model_type}_t{target_class}','images')
                os.makedirs(output_dir, exist_ok=True)
                img_idx = i * args.batch_size + j
                out_path = os.path.join(output_dir, f'{target_class}_{img_idx}.png')
                g_img.save(out_path)

    print("Done generating adversarial images.")

if __name__ == "__main__":
    main()
