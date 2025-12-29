# import argparse
# import os

# import torch
# import numpy as np
# from torchvision import datasets, transforms

# from utils import *
# from models.generator import CrossAttenGenerator


# def main():
#     parser = argparse.ArgumentParser(description='CLIP-based Generative Networks – Evaluation')
#     parser.add_argument('--dataset', type=str, default='cifar10',choices=['imagenet', 'cifar10'],help='Dataset type')
#     parser.add_argument('--data_dir', type=str,default=r'D:\cifar-10-python\cifar10_png\test', help='Path to clean test images (ImageFolder style)')
#     parser.add_argument('--is_nips', action='store_true', default=False,help='Evaluation on NIPS data (special label mapping)')
#     parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
#     parser.add_argument('--eps', type=int, default=16, help='Perturbation budget (in 0–255 scale)')
#     parser.add_argument('--model_type', type=str, default='cifar10_vgg19_bn',help='Source model used to train the generator (for naming only)')
#     parser.add_argument('--load_path', type=str, default=r'E:\PhD-Software-Engineering\Code\CGNC_LOW_Final\checkpoints_cifar10_ins\cifar10_vgg19_bn\model-9.pth', help='Path to generator checkpoint')
#     parser.add_argument('--label_flag', type=str, default='C5', help='Label set: CIFAR-10: C3,C5,C8,ALL; ImageNet: N8,C20,C50,C100,C200,...')
#     parser.add_argument('--nz', type=int, default=16, help='Latent/cond dim for generator')
#     parser.add_argument('--finetune', action='store_true', help='Finetune/evaluate single-class attack')
#     parser.add_argument('--finetune_class', type=int,help='Class id to be finetuned (global label)')
#     parser.add_argument('--save_dir', type=str, default='results_cifar10_ins',help='Directory to save adversarial images')

#     # ---- prompt-related flags ----
#     parser.add_argument('--prompt_mode', type=str, default='instance',choices=['precomputed', 'learnable', 'instance'],help='How to obtain CLIP text features')
#     parser.add_argument('--text_feature_path', type=str, default=None,help='Path to precomputed text-feature .pth (for prompt_mode=precomputed)')
#     parser.add_argument('--clip_backbone', type=str, default='ViT-B/16',help='CLIP backbone (for learnable prompts)')
#     parser.add_argument('--ctx_dim', type=int, default=512, help='Dim of learnable prompt context (for learnable prompts)')
#     parser.add_argument('--prompt_ckpt', type=str, default=r'E:\PhD-Software-Engineering\Code\CGNC_LOW_Final\checkpoints_cifar10_ins\cifar10_vgg19_bn\prompt-9.pth',help='Checkpoint of LearnablePrompt (for prompt_mode=learnable)')

#     parser.add_argument('--n_ctx', type=int, default=16,help='Number of context tokens per prompt (AdvancedLearnablePrompt)')
#     parser.add_argument('--n_prompts', type=int, default=8,help='Number of prompts per class (AdvancedLearnablePrompt)')

#     parser.add_argument('--k',type=int, default=2,help='Gaussian radius k for low-pass filter (kernel size = 4k+1, sigma = k). ''If 0, use original kernel_size=7, sigma=2.')

#     args = parser.parse_args() 
#     print(args)

#     eps = args.eps / 255.0

#     k_for_gen = args.k if args.k > 0 else None

#     # GPU
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # ---- Build generator (same as before) ----
#     if args.model_type == 'incv3' and args.dataset == 'imagenet':
#         scale_size, img_size = 300, 299
#         netG = CrossAttenGenerator(inception=True, nz=args.nz, k=k_for_gen, device=device)
#     else:
#         # CIFAR-10: (32,32); ImageNet: (256,224)
#         scale_size, img_size = (32, 32) if args.dataset == 'cifar10' else (256, 224)
#         netG = CrossAttenGenerator(nz=args.nz, k=k_for_gen, device=device)

#     # Load Generator
#     print(f"[EVAL] Loading generator from {args.load_path}")
#     state_dict = torch.load(args.load_path, map_location=device)
#     netG.load_state_dict(state_dict)
#     netG = netG.to(device)
#     netG.eval()

#     # ---- Data ----
#     data_transform = transforms.Compose([
#         transforms.Resize(scale_size),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#     ])

#     test_set = datasets.ImageFolder(args.data_dir, data_transform)

#     # Fix labels if needed
#     if args.is_nips:
#         test_set = fix_labels_nips(args, test_set, pytorch=True)
#     else:
#         test_set = fix_labels(args, test_set)

#     test_loader = torch.utils.data.DataLoader(
#         test_set,
#         batch_size=args.batch_size,
#         shuffle=False
#     )

#     # ---- Choose global class IDs to attack ----
#     if args.finetune:
#         class_ids = np.array([args.finetune_class])
#     else:
#         class_ids = get_classes(args.label_flag, dataset=args.dataset)
#     print(f"[EVAL] Class IDs for attack: {class_ids}")

#     # ---- Prompt / text condition setup ----
#     prompt_learner = None
#     global_to_local = None

#     if args.prompt_mode == 'precomputed':
#         # default paths if not given
#         if args.text_feature_path is None:
#             if args.dataset == 'cifar10':
#                 # e.g. created by build_text_features.py
#                 args.text_feature_path = 'cifar10_text_feature_multi_prompt.pth'
#             else:
#                 args.text_feature_path = 'imagenet_text_feature_multi_prompt.pth'

#         print(f"[EVAL] Loading precomputed text features from: {args.text_feature_path}")
#         text_cond_dict = torch.load(
#             args.text_feature_path,
#             map_location=device
#         )
#         # ensure on device
#         for k, v in text_cond_dict.items():
#             if isinstance(v, torch.Tensor):
#                 text_cond_dict[k] = v.to(device)
#     else:
#         # Learnable prompts: rebuild the same subset as in training
#         from prompt_learner import AdvancedLearnablePrompt, InstanceConditionedPrompt

#         class_index = getClassIndex(args.dataset)
#         # sort for deterministic mapping
#         used_ids = np.array(sorted(class_ids.tolist()))
#         classnames = [class_index[int(i)][1] for i in used_ids]

#         if args.prompt_mode == 'learnable':
#             print("[EVAL] Using LearnablePrompt for classes:")
#             for gid, name in zip(used_ids, classnames):
#                 print(f"  global id {int(gid):4d} -> '{name}'")

#             prompt_learner = AdvancedLearnablePrompt(classnames=classnames,clip_backbone=args.clip_backbone,device=device,ctx_dim=args.ctx_dim,n_ctx=args.n_ctx,n_prompts=args.n_prompts,).to(device)

#         elif args.prompt_mode == 'instance':
#             print("[EVAL] Using InstanceConditionedPrompt (CoCoOp-style) for classes:")
#             for gid, name in zip(class_ids, classnames):
#                 print(f"  global id {gid:4d} -> '{name}'")

#             prompt_learner = InstanceConditionedPrompt(
#                     classnames=classnames,
#                     clip_backbone=args.clip_backbone,
#                     device=device,
#                     n_ctx=args.n_ctx,
#                 ).to(device)
#         else:
#             raise ValueError(f"Unknown prompt_mode: {args.prompt_mode}")

#         if args.prompt_ckpt is None:
#             raise ValueError("You must provide --prompt_ckpt when prompt_mode='learnable'")

#         print(f"[EVAL] Loading LearnablePrompt checkpoint from: {args.prompt_ckpt}")
#         prompt_state = torch.load(args.prompt_ckpt, map_location=device)
#         prompt_learner.load_state_dict(prompt_state)
#         prompt_learner.eval()

#         # Map global label -> local index for prompt_learner
#         global_to_local = {int(gid): idx for idx, gid in enumerate(used_ids)}
#         text_cond_dict = None  # not used in this mode

#     # ---- Generate & save adversarial images ----
#     netG.eval()
#     if prompt_learner is not None:
#         prompt_learner.eval()

#     for idx, target_class in enumerate(class_ids):
#         target_class = int(target_class)
#         print(f"[EVAL] Generating adv examples for target class {target_class} "
#               f"({idx+1}/{len(class_ids)})")

#         for i, (img, _) in enumerate(test_loader):
#             img = img.to(device)
#             b = img.size(0)

#             # Build text condition 'cond' depending on prompt mode
#             if args.prompt_mode == 'precomputed':
#                 base_feat = text_cond_dict[target_class]  # [D]
#                 cond = base_feat.unsqueeze(0).expand(b, -1)  # [B, D]
#             else:
#                 # learnable prompts
#                 local_idx = torch.full((b,),global_to_local[target_class],dtype=torch.long,device=device)
#                 cond = prompt_learner(local_idx)  # [B, D]

#             adv = netG(img, cond, eps=eps)

#             # ensure within epsilon-ball & valid pixel range
#             adv = torch.min(torch.max(adv, img - eps), img + eps)
#             adv = torch.clamp(adv, 0.0, 1.0)

#             save_imgs = adv.detach().cpu()
#             for j in range(len(save_imgs)):
#                 g_img = transforms.ToPILImage('RGB')(save_imgs[j])

#                 # same directory pattern as your inference code expects
#                 output_dir = os.path.join(
#                     args.save_dir,
#                     'gan_n8',  # keep as is for compatibility
#                     f'{args.model_type}_t{target_class}',
#                     'images'
#                 )
#                 os.makedirs(output_dir, exist_ok=True)

#                 img_idx = i * args.batch_size + j
#                 out_path = os.path.join(output_dir, f'{target_class}_{img_idx}.png')
#                 g_img.save(out_path)

#     print("[EVAL] Done generating adversarial images.")


# if __name__ == "__main__":
#     main()

import argparse
import os

import torch
import numpy as np
from torchvision import datasets, transforms

from utils import *
from models.generator import Generator


def main():
    parser = argparse.ArgumentParser(description='Data-Efficient Multi-Target Generative Attack with Learnable Prompts- Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_dir', type=str,default='cifar10/test')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--eps', type=int, default=16)
    parser.add_argument('--model_type', type=str, default='cifar10_resnet56',help='cifar10_vgg19_bn')
    parser.add_argument('--load_path', type=str, default='model-9.pth', help='Path to generator checkpoint')
    parser.add_argument('--label_flag', type=str, default='N8', help='CIFAR-10: C3,C5,C8,ALL')
    parser.add_argument('--nz', type=int, default=16, help='Latent/cond dim for generator')
    parser.add_argument('--save_dir', type=str, default='results_cifar10')
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

    # ---- Build generator ----
    scale_size, img_size = (32, 32) 
    netG = Generator(nz=args.nz, k=k_for_gen, device=device)

    # Load Generator
    print(f"[Loading generator from {args.load_path}")
    state_dict = torch.load(args.load_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG = netG.to(device)
    netG.eval()

    # ---- Data ----
    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    test_set = datasets.ImageFolder(args.data_dir, data_transform)

    # Fix labels if needed
    test_set = fix_labels(args, test_set)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)

    class_ids = get_classes(args.label_flag, dataset=args.dataset)
    print(f"Class IDs for attack: {class_ids}")

    # ---- Prompt / text condition setup ----
    prompt_learner = None
    global_to_local = None

    if args.prompt_mode == 'precomputed':
        args.text_feature_path = 'cifar10_text_feature_multi_prompt.pth'
        print(f"Loading precomputed text features from: {text_feature_path}")
        text_cond_dict = torch.load(args.text_feature_path, map_location=device)
        for k, v in text_cond_dict.items():
            if isinstance(v, torch.Tensor):
                text_cond_dict[k] = v.to(device)
    else:
        from prompt_learner import LearnablePrompt

        class_index = getClassIndex(args.dataset)
        used_ids = np.array(sorted(class_ids.tolist()))
        classnames = [class_index[int(i)][1] for i in used_ids]

        if args.prompt_mode == 'learnable':
            print("Using LearnablePrompt (class-only) for classes:")
            for gid, name in zip(used_ids, classnames):
                print(f"  global id {int(gid):4d} -> '{name}'")
            prompt_learner = LearnablePrompt(classnames=classnames,clip_backbone=args.clip_backbone,device=device,ctx_dim=args.ctx_dim, ).to(device)

        print(f"Loading prompt learner checkpoint from: {args.prompt_ckpt}")
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

            elif args.prompt_mode == 'learnable':
                local_idx = torch.full(
                    (b,),
                    global_to_local[target_class],
                    dtype=torch.long,
                    device=device
                )
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
