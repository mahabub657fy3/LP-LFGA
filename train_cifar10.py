from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from models.generator import Generator
from image_transformer import rotation
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
Image.MAX_IMAGE_PIXELS = None
import torch.optim as optim
import random


def main():
    parser = argparse.ArgumentParser(description='Clip-based Generative Networks')
    # ---- Basic setup ----
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_dir', default='cifar10/train', help='imagenet')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--eps', type=int, default=16, help='Perturbation budget')
    parser.add_argument('--model_type', type=str, default='cifar10_resnet56', help='Source model: cifar10_vgg19_bn')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--label_flag', type=str, default='C5', help='Label set: CIFAR-10: C3,C5,C8,ALL')
    parser.add_argument('--nz', type=int, default=16, help='nz')
    parser.add_argument('--save_dir', type=str, default='checkpoints_cifar10', help='Dictionary to save the model')
    parser.add_argument('--load_path', type=str, help='Path to checkpoint')
    parser.add_argument('--prompt_mode', type=str, default='learnable',choices=['precomputed', 'learnable'],help='How to obtain CLIP text features')
    parser.add_argument('--clip_backbone', type=str, default='ViT-B/16')
    parser.add_argument('--ctx_dim', type=int, default=512)
    parser.add_argument('--prompt_ckpt', type=str, default=None)
    parser.add_argument('--k',type=int, default=2)

    args = parser.parse_args()
    print(args)

    # ---- Basic numeric setup ----
    eps = args.eps / 255.0
    k_for_gen = args.k if args.k > 0 else None

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure save dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Input dimension and generator
    scale_size, img_size = (32, 32)
    netG = Generator(nz=args.nz, k=k_for_gen, device=device)

    if args.start_epoch > 0:
        netG.load_state_dict(torch.load(args.load_path, map_location=device))
    netG = netG.to(device)

    # Data
    train_set = get_data(args.train_dir, scale_size, img_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # CIFAR-10 surrogate
    model = load_cifar10_model(args.model_type, device)
    norm_fn = normalize_cifar10
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    # ---- Class subset ----
    label_set = get_classes(args.label_flag, dataset=args.dataset)
    print(f"Label set: {label_set}")

    # ---- Loss ----
    criterion = nn.CrossEntropyLoss()

    # ---- Prompt / text condition setup ----
    prompt_learner = None
    global_to_local = None

    if args.prompt_mode == 'precomputed':
        text_feature_path = 'cifar10_text_feature_multi_prompt.pth'
        print(f"Loading precomputed text features from: {text_feature_path}")
        text_cond_dict = torch.load(args.text_feature_path,map_location=device)
        
        for k, v in text_cond_dict.items():
            if isinstance(v, torch.Tensor):
                text_cond_dict[k] = v.to(device)
    else:
        from prompt_learner import LearnablePrompt

        class_index = getClassIndex(args.dataset)
        class_ids = np.array(sorted(label_set.tolist()))
        classnames = [class_index[int(i)][1] for i in class_ids]

        if args.prompt_mode == 'learnable':
            print("Using LearnablePrompt (class-only) for classes:")
            for gid, name in zip(class_ids, classnames):
                print(f"global id {gid:4d} -> '{name}'")

            prompt_learner = LearnablePrompt(classnames=classnames,clip_backbone=args.clip_backbone,device=device,ctx_dim=args.ctx_dim, ).to(device)
                
        # Optional resume for prompt_learner
        if args.start_epoch > 0 and args.prompt_ckpt is not None:
            print(f"[TRAIN] Loading LearnablePrompt from: {args.prompt_ckpt}")
            pl_state = torch.load(args.prompt_ckpt, map_location=device)
            prompt_learner.load_state_dict(pl_state)

        # Map global label -> local index used by prompt_learner
        global_to_local = {int(gid): idx for idx, gid in enumerate(class_ids)}
        text_cond_dict = None  # not used in this mode

    # ---- Optional resume for generator ----
    if args.start_epoch > 0 and args.load_path is not None:
        print(f"[TRAIN] Loading generator from: {args.load_path}")
        g_state = torch.load(args.load_path, map_location=device)
        netG.load_state_dict(g_state)

    # ---- Optimizer (generator + prompt_learner) ----
    if args.prompt_mode in ['learnable'] and prompt_learner is not None:
        params = list(netG.parameters()) + list(prompt_learner.parameters())
    else:
        params = netG.parameters()

    optimG = optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))

    # save dir
    save_dir = os.path.join(args.save_dir, args.model_type)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        netG.train()
        if prompt_learner is not None:
            prompt_learner.train()
        running_loss = 0.0

        for i, (imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            img = imgs[0].to(device)       
            img_rot = rotation(img)[0]      
            img_aug = imgs[1].to(device)    

            np.random.shuffle(label_set)
            label = np.random.choice(label_set, img.size(0))
            label = torch.from_numpy(label).long().to(device)

            # ---- Build text condition ----
            if args.prompt_mode == 'precomputed':
                cond_list = [text_cond_dict[int(j)] for j in label]
                cond = torch.stack(cond_list, dim=0).to(device)   # [B, D]

            elif args.prompt_mode == 'learnable':
                local_idx = torch.tensor(
                    [global_to_local[int(j)] for j in label],
                    dtype=torch.long,
                    device=device
                )
                cond = prompt_learner(local_idx)                  # [B, D]
            optimG.zero_grad()

            adv = netG(input=img,     cond=cond, eps=eps)
            adv_rot = netG(input=img_rot, cond=cond, eps=eps)
            adv_aug = netG(input=img_aug, cond=cond, eps=eps)

            adv     = torch.clamp(adv,     0.0, 1.0)
            adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
            adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

            adv_out     = model(norm_fn(adv))
            adv_rot_out = model(norm_fn(adv_rot))
            adv_aug_out = model(norm_fn(adv_aug))

            loss = criterion(adv_out, label) + criterion(adv_rot_out, label) + criterion(adv_aug_out, label)
            loss.backward()
            optimG.step()

            if i % 100 == 99:
                print('Epoch: {} \t Batch: {}/{} \t loss: {:.5f}'.format(epoch, i, len(train_loader), running_loss / 100))
                running_loss = 0
            running_loss += abs(loss.item())

        if epoch >= args.start_epoch:
            if torch.cuda.device_count() > 1:
                torch.save(netG.module.state_dict(), '{}/model-{}.pth'.format(save_dir, epoch))
            else:
                torch.save(netG.state_dict(), '{}/model-{}.pth'.format(save_dir, epoch))

            if args.prompt_mode in ['learnable'] and prompt_learner is not None:
                torch.save(prompt_learner.state_dict(), f'{save_dir}/prompt-{epoch}.pth')


if __name__ == "__main__":
    main()