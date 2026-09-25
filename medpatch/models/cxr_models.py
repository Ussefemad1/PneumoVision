
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
from .loss import RankingLoss, CosineLoss
import torch.nn.functional as F

class CXRModels(nn.Module):

    def __init__(self, args, device='cpu'):
	
        super(CXRModels, self).__init__()
        self.args = args
        self.device = device
        self.vision_backbone = getattr(torchvision.models, self.args.vision_backbone)(pretrained=self.args.pretrained)
        classifiers = [ 'classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
            break
        self.bce_loss = torch.nn.BCELoss(size_average=True)
        self.classifier = nn.Sequential(nn.Linear(d_visual, self.args.vision_num_classes))
        self.feats_dim = d_visual
       

    def forward(self, x, labels=None, n_crops=0, bs=16):
        lossvalue_bce = torch.zeros(1).to(self.device)

        visual_feats = self.vision_backbone(x)
        preds = self.classifier(visual_feats)

        preds = torch.sigmoid(preds)

        if n_crops > 0:
            preds = preds.view(bs, n_crops, -1).mean(1)
        if labels is not None:
            lossvalue_bce = self.bce_loss(preds, labels)

        return preds, lossvalue_bce, visual_feats
    

def _describe(model, label, extra=''):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=' * 66)
    print(f"Building CXR encoder: {label}")
    print(f"  class          : {type(model).__name__}")
    print(f"  parameters     : {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    print(f"  feats_dim      : {getattr(model, 'feats_dim', 'n/a')}")
    if extra:
        print(f"  {extra}")
    print('=' * 66)
    return model


def CXR_encoder(args, device='cpu'):
    """Build the CXR encoder named by --cxr_encoder.

    Defect 6: this used to be a bare alias for CXRModels, so --cxr_encoder was
    read only as a boolean gate -- `if args.cxr_encoder is not None` -- and its
    value was discarded. Passing 'vit_small_patch16_384' silently built
    torchvision densenet121 (the --vision-backbone default) instead of the
    paper's ViT, producing a plausible but meaningless AUROC.

    Now the value selects the architecture:

      a timm model name  -> CXRTransformer, [B, N, D] patch tokens (the paper's
                            setup: 384/16 = 24, so 24^2 + 1 CLS = 577 tokens,
                            matching max_seq_len = 578 in fusion.py)
      anything else      -> CXRModels, a torchvision backbone named by
                            --vision-backbone, producing a pooled [B, D] vector
                            with no token dimension

    Construction mirrors DHF_trainer.py, which already built CXRTransformer
    correctly.
    """
    import timm
    from .cxr_encoder import CXRTransformer

    name = getattr(args, 'cxr_encoder', None)

    # CXRTransformer.forward does `b, n, _ = x.shape`, so it requires a model
    # whose forward_features returns a 3-D token sequence. timm.is_model() alone
    # is not enough -- it returns True for densenet121 too, whose
    # forward_features gives a 4-D feature map and would crash here. Restrict to
    # the token-based families.
    TOKEN_MODEL_PREFIXES = ('vit_', 'deit_', 'deit3_', 'beit_', 'eva_', 'eva02_')
    is_token_model = bool(name) and name.startswith(TOKEN_MODEL_PREFIXES)

    if name and is_token_model and timm.is_model(name):
        image_size = getattr(args, 'image_size', 384)
        patch_size = getattr(args, 'patch_size', 16)
        model = CXRTransformer(
            model_name=name,
            image_size=image_size,
            patch_size=patch_size,
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            emb_dropout=0.0,
            dim_head=128,
        ).to(device)

        backbone = sum(p.numel() for p in model.feature_extractor.parameters())
        tokens = (image_size // patch_size) ** 2 + 1
        return _describe(
            model, name,
            extra=(f"backbone       : {backbone/1e6:.1f}M (the rest is the unused "
                   f"internal transformer/pos_embedding -- see note below)\n"
                   f"  tokens/image   : {tokens} "
                   f"({image_size}/{patch_size} = {image_size//patch_size}, "
                   f"{image_size//patch_size}^2 + 1 CLS)")
        )

    if name:
        if timm.is_model(name) and not is_token_model:
            reason = ("is a timm model but not a token-based one; CXRTransformer "
                      "needs a 3-D token output")
        else:
            reason = "is not a recognised timm token model"
        print(f"[cxr] --cxr_encoder '{name}' {reason}. Falling back to "
              f"torchvision --vision-backbone "
              f"'{getattr(args, 'vision_backbone', '?')}'.")

    return _describe(
        CXRModels(args, device),
        getattr(args, 'vision_backbone', 'densenet121'),
        extra="pooled [B, D] output -- no token dimension, so --use_cls_token has no effect"
    )
