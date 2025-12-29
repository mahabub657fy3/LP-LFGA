import torch
import torch.nn as nn
import clip

class LearnablePrompt(nn.Module):
    """
    CoOp-style prompt learner that operates in CLIP text-embedding space.
    - CLIP is frozen.
    - For each class (within your label subset), we keep a learnable context vector.
    - Final text feature = normalized(base_CLIP_feature + projected_context).
    """
    def __init__(self,
                 classnames,
                 clip_backbone="ViT-B/16",
                 device="cuda",
                 ctx_dim=512):
        super().__init__()
        self.device = device
        self.classnames = list(classnames)
        self.n_cls = len(self.classnames)
        self.ctx_dim = ctx_dim

        # Load and freeze CLIP text encoder
        self.clip_model, _ = clip.load(clip_backbone, device=device, jit=False)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Precompute base text features with a simple template
        with torch.no_grad():
            texts = [f"a photo of a {name}" for name in self.classnames]
            tokenized = clip.tokenize(texts).to(device)
            base = self.clip_model.encode_text(tokenized)  # [n_cls, D]
            base = base / base.norm(dim=-1, keepdim=True)
        self.register_buffer("base_features", base)  # not trainable

        # Learnable context vectors, one per class
        self.prompt_ctx = nn.Parameter(torch.zeros(self.n_cls, ctx_dim))
        nn.init.normal_(self.prompt_ctx, std=0.02)

        # Projection into CLIP text-embedding dim (usually 512)
        embed_dim = base.shape[1]
        if ctx_dim != embed_dim:
            self.ctx_proj = nn.Linear(ctx_dim, embed_dim)
        else:
            self.ctx_proj = nn.Identity()

    def forward(self, class_indices):
        """
        class_indices: LongTensor shape [B]
            indices relative to `classnames` list (0 .. n_cls-1).
        returns:
            text_features: FloatTensor [B, D] (L2-normalized)
        """
        if not torch.is_tensor(class_indices):
            class_indices = torch.tensor(class_indices, device=self.base_features.device)
        class_indices = class_indices.to(self.base_features.device)

        ctx = self.prompt_ctx[class_indices]          # [B, ctx_dim]
        ctx_feat = self.ctx_proj(ctx)                 # [B, D]
        base_feat = self.base_features[class_indices] # [B, D]
        feat = base_feat + ctx_feat
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

