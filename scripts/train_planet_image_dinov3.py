#!/usr/bin/env python
"""Train a DINOv3 crown-event segmenter on the Planet Mask R-CNN chips.

The backbone is a DINOv3 ViT-L/16 from Hugging Face, by default the
satellite-pretrained one (``facebook/dinov3-vitl16-pretrain-sat493m``, SAT-493M
Maxar RGB at 0.6 m), and ``--backbone facebook/dinov3-vitl16-pretrain-lvd1689m``
is the web-image control. Both are gated: accept the licence on the model page
and log in with ``hf auth login`` once, or point HF_HUB_CACHE at a copied hub
cache and set HF_HUB_OFFLINE=1 (what the gattaca2 configs do).

Inputs, labels, windows, folds and cloud handling are those of
train_planet_image_maskrcnn.py (its PlanetMaskRCNNDataset): RGB with the
per-image or --fixed-stretch stretch, the same ColorJitter, and both 512 x 512
windows of each chip under --folds/--fold. The model input is normalized with
the backbone's own mean and std.

Model: patch tokens of four backbone layers (6, 12, 18, 24 of 24, each through
the final LayerNorm, as DINOv3's get_intermediate_layers) are projected to
--decoder-dim channels, resampled to 1/4 of the input (128 x 128, i.e. 3 m,
Planet's native pixel), concatenated with a small convolutional stem on the
input image at the same resolution (--image-stem, default on, because 16 px =
12 m tokens are coarse for crown edges; its value is untested, see
docs/planet_temporal_generalization.md, "DINOv3 backbone"), fused by two 3 x 3
conv blocks (GroupNorm), and upsampled to a per-pixel event logit. With --finetune-blocks 0 (default) the backbone is
frozen and only the decoder trains. With N > 0 the last N blocks and the
final norm are also trained, at --backbone-lr.

Target: the union of the window's labelled instances (crowns larger than
--min-instance-size that do not touch the window border, as the Mask R-CNN
targets). Labelled crowns that are dropped by that rule, and OCM-cloudy pixels
under --ocm-masks, are ignored in the loss (BCE + soft Dice).

Instances for scoring (predict_instances): 8-connected components of
probability >= the mask threshold, larger than --min-pred-size pixels, scored
by their mean probability, and dropped when their box centre is cloudy. This
is the rule filtered_predictions applies to Mask R-CNN output, so
eval_temporal_cv.py scores both model kinds the same way. It recognizes these
checkpoints by their 'kind' key.

Checkpoints (epoch_NNN.pth, last_model.pth) hold the params and the trainable
weights only (decoder, plus fine-tuned blocks). The frozen backbone is
reloaded from Hugging Face.

Typical usage (one temporal fold):
    python scripts/train_planet_image_dinov3.py <chip_dir> runs/dinov3_fold0 \\
        --folds config/temporal_folds_20261012.csv --fold 0 \\
        --ocm-masks --min-instance-size 64 --num-epochs 40 \\
        --save-epochs 1,2,3,5,8,12,17,25,40
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede torch import
import json
import random
import time

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import masks_to_boxes

from train_planet_image_maskrcnn import (
    PlanetMaskRCNNDataset,
    _lookup_clear_at_centers,
    collate_fn,
    get_split,
    load_stretch,
    read_fold_chips,
)
from util import select_device

SAT = 'facebook/dinov3-vitl16-pretrain-sat493m'
WEB = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
# Normalization of each backbone's preprocessor_config.json.
BACKBONE_NORM = {
    SAT: ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),
    WEB: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
TAP_LAYERS = (5, 11, 17, 23)  # 0-based, of 24


class DinoSegmenter(nn.Module):

    def __init__(self, backbone_id, finetune_blocks=0, decoder_dim=256,
                 tap_layers=TAP_LAYERS, image_stem=True):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(backbone_id)
        cfg = self.backbone.config
        self.patch = cfg.patch_size
        self.n_prefix = 1 + cfg.num_register_tokens  # CLS + registers
        self.tap_layers = tuple(tap_layers)
        n_layers = len(self.backbone.layer)
        self.first_trainable = n_layers - finetune_blocks
        for p in self.backbone.parameters():
            p.requires_grad = False
        if finetune_blocks:
            for blk in self.backbone.layer[self.first_trainable:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
        mean, std = BACKBONE_NORM[backbone_id]
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

        hidden = cfg.hidden_size
        self.proj = nn.ModuleList(nn.Conv2d(hidden, decoder_dim, 1) for _ in self.tap_layers)

        # GroupNorm, not BatchNorm: with batches of 2-4 windows, BatchNorm's
        # running statistics made held-out IoU drift while the loss fell.
        def block(cin, cout, stride=1):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                                 nn.GroupNorm(min(32, cout // 4), cout), nn.ReLU(inplace=True))
        # Image features at 1/4 resolution, for edges finer than a token.
        self.stem = (nn.Sequential(block(3, 32), block(32, 64, 2), block(64, 64, 2))
                     if image_stem else None)
        n_in = decoder_dim * len(self.tap_layers) + (64 if image_stem else 0)
        self.fuse = nn.Sequential(block(n_in, decoder_dim),
                                  block(decoder_dim, decoder_dim // 2))
        self.head = nn.Conv2d(decoder_dim // 2, 1, 1)

    def features(self, x):
        """Patch-token maps (B, hidden, H/16, W/16) at the tapped layers."""
        b, _, h, w = x.shape
        bb = self.backbone
        x = (x - self.mean) / self.std
        # Frozen blocks run without autograd, so a frozen backbone costs no
        # activation memory in training.
        grad = torch.is_grad_enabled()
        with torch.set_grad_enabled(False):
            tokens = bb.embeddings(x)
            pos = bb.rope_embeddings(x)
        maps = []
        for i, layer in enumerate(bb.layer):
            with torch.set_grad_enabled(grad and i >= self.first_trainable):
                tokens = layer(tokens, position_embeddings=pos)
                if i in self.tap_layers:
                    t = bb.norm(tokens)[:, self.n_prefix:]
                    maps.append(t.transpose(1, 2).reshape(b, -1, h // self.patch,
                                                          w // self.patch))
        return maps

    def forward(self, x):
        """Per-pixel event logits (B, H, W) for images (B, 3, H, W) in [0, 1]."""
        h, w = x.shape[-2:]
        if h % self.patch or w % self.patch:
            raise ValueError(f'input {h}x{w} is not a multiple of {self.patch}')
        quarter = (h // 4, w // 4)
        parts = [F.interpolate(p(m), size=quarter, mode='bilinear', align_corners=False)
                 for p, m in zip(self.proj, self.features(x))]
        if self.stem is not None:
            parts.append(self.stem((x - self.mean) / self.std))
        f = torch.cat(parts, dim=1)
        logits = self.head(self.fuse(f))
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)[:, 0]

    def trainable_state_dict(self):
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith('backbone.') or k in trainable}


def model_from_checkpoint(ckpt, device):
    """Rebuild a DinoSegmenter from a checkpoint written by this script."""
    p = ckpt['params']
    model = DinoSegmenter(p['backbone'], p['finetune_blocks'], p['decoder_dim'],
                          p['tap_layers'], p.get('image_stem', False))
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if unexpected or any(not k.startswith('backbone.') for k in missing):
        raise RuntimeError(f'checkpoint mismatch: missing {missing[:5]}, '
                           f'unexpected {unexpected[:5]}')
    model.min_pred_size = p['min_pred_size']
    return model.to(device).eval()


@torch.no_grad()
def predict_instances(model, imgs, targets, device, score_thresh=0.5,
                      use_ocm_masks=False, min_pred_size=None):
    """filtered_predictions' output format for a DinoSegmenter: per image a
    dict of boolean instance masks (connected components of probability >=
    score_thresh), mean-probability scores, labels and boxes, with
    components whose box centre is cloudy dropped."""
    if min_pred_size is None:
        min_pred_size = getattr(model, 'min_pred_size', 4)
    probs = torch.sigmoid(model(torch.stack(imgs).to(device)))
    clear_masks = ([t['clear_mask'].to(device).bool() for t in targets]
                   if use_ocm_masks else None)
    preds = []
    for i, prob in enumerate(probs):
        lab, n = ndimage.label((prob >= score_thresh).cpu().numpy(),
                               structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        ids = [j for j in range(1, n + 1) if sizes[j] > min_pred_size]
        if ids:
            lab_t = torch.from_numpy(lab).to(device)
            masks = torch.stack([lab_t == j for j in ids])
            scores = torch.stack([prob[m].mean() for m in masks])
            boxes = masks_to_boxes(masks).float()
        else:
            masks = torch.zeros((0, *prob.shape), dtype=torch.bool, device=device)
            scores = torch.zeros((0,), device=device)
            boxes = torch.zeros((0, 4), device=device)
        if clear_masks is not None and boxes.shape[0]:
            keep = _lookup_clear_at_centers(clear_masks, boxes, i)
            masks, scores, boxes = masks[keep], scores[keep], boxes[keep]
        preds.append({'masks': masks, 'scores': scores, 'boxes': boxes,
                      'labels': torch.ones((masks.shape[0],), dtype=torch.int64,
                                           device=device)})
    return preds


class SemanticWindows(Dataset):
    """PlanetMaskRCNNDataset windows as (image, target, ignore) for semantic
    training: target is the union of the kept instances; ignore marks
    labelled pixels the instance rule dropped, and cloudy pixels."""

    def __init__(self, base, use_ocm_masks):
        self.base = base
        self.use_ocm_masks = use_ocm_masks

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, tgt = self.base[idx]
        h, w = img.shape[-2:]
        target = (tgt['masks'].any(dim=0) if tgt['masks'].shape[0]
                  else torch.zeros((h, w), dtype=torch.bool))
        file_idx, split = self.base.items[idx]
        raw = np.array(Image.open(self.base.mask_files[file_idx])) == 255
        _, raw = get_split(raw, raw, split, self.base.size)
        ignore = torch.from_numpy(raw.copy()) & ~target
        if self.use_ocm_masks:
            ignore |= ~tgt['clear_mask'].bool()
        return img, target.float(), ignore.bool()


def bce_dice_loss(logits, target, ignore):
    valid = (~ignore.bool()).float()  # a uint8 ignore would make ~ 254/255
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    prob = torch.sigmoid(logits) * valid
    t = target * valid
    inter = (prob * t).sum(dim=(1, 2))
    dice = 1 - (2 * inter + 1) / (prob.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def window_iou(model, loader, device, thresh=0.5):
    """Binary IoU over clear (non-ignored) pixels of a semantic loader."""
    inter = union = 0
    for img, target, ignore in loader:
        prob = torch.sigmoid(model(img.to(device)))
        keep = ~ignore.to(device).bool()
        pred = (prob >= thresh) & keep
        t = target.to(device).bool() & keep
        inter += (pred & t).sum().item()
        union += (pred | t).sum().item()
    return inter / union if union else float('nan')


@click.command()
@click.argument('imagedir')
@click.argument('outputdir')
@click.option('--backbone', default=SAT, show_default=True,
              type=click.Choice(sorted(BACKBONE_NORM)))
@click.option('--finetune-blocks', default=0, show_default=True,
              help='Train the last N backbone blocks too (0: frozen backbone).')
@click.option('--decoder-dim', default=256, show_default=True)
@click.option('--image-stem/--no-image-stem', default=True, show_default=True,
              help='Add a 1/4-resolution conv stem on the input image to the decoder.')
@click.option('--num-epochs', default=40, show_default=True)
@click.option('--batch-size', default=4, show_default=True)
@click.option('--lr', default=1e-3, show_default=True, help='Decoder learning rate.')
@click.option('--backbone-lr', default=1e-5, show_default=True,
              help='Learning rate of fine-tuned backbone blocks.')
@click.option('--weight-decay', default=1e-4, show_default=True)
@click.option('--size', default=512, show_default=True)
@click.option('--min-instance-size', default=4, show_default=True)
@click.option('--min-pred-size', default=4, show_default=True,
              help='Predicted components of at most this many pixels are dropped.')
@click.option('--ocm-masks/--no-ocm-masks', 'use_ocm_masks', default=False)
@click.option('--color-jitter/--no-color-jitter', 'color_jitter', default=True)
@click.option('--fixed-stretch', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--folds', 'folds_csv', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--fold', default=None)
@click.option('--chips', 'chip_list', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Train (and test) only on the chips listed in this file, one per '
                   'line, without --folds; for overfitting checks.')
@click.option('--save-epochs', default=None,
              help='Comma-separated epochs to write epoch_NNN.pth for; '
                   'last_model.pth is always written.')
@click.option('--seed', default=None, type=int)
def main(imagedir, outputdir, backbone, finetune_blocks, decoder_dim, image_stem, num_epochs,
         batch_size, lr, backbone_lr, weight_decay, size, min_instance_size,
         min_pred_size, use_ocm_masks, color_jitter, fixed_stretch, folds_csv, fold,
         chip_list, save_epochs, seed):
    if (folds_csv is None) != (fold is None):
        raise click.UsageError('--folds and --fold must be given together')
    os.makedirs(outputdir, exist_ok=True)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if folds_csv is not None:
        train_split = test_split = ('left', 'right')
        train_chips, test_chips = read_fold_chips(folds_csv, fold)
    elif chip_list is not None:
        with open(chip_list) as fh:
            train_chips = test_chips = [l.strip() for l in fh if l.strip()]
        train_split = test_split = ('left', 'right')
    else:
        train_split, test_split = 'left', 'right'
        train_chips = test_chips = None
    save_epochs = ({int(e) for e in save_epochs.split(',') if e.strip()}
                   if save_epochs else set())
    stretch = load_stretch(fixed_stretch)
    channel_kinds = ['red', 'green', 'blue']

    params = {
        'kind': 'dinov3', 'backbone': backbone, 'finetune_blocks': finetune_blocks,
        'decoder_dim': decoder_dim, 'tap_layers': list(TAP_LAYERS),
        'image_stem': image_stem, 'lr': lr,
        'backbone_lr': backbone_lr, 'weight_decay': weight_decay,
        'batch_size': batch_size, 'num_epochs': num_epochs, 'size': size,
        'min_instance_size': min_instance_size, 'min_pred_size': min_pred_size,
        'use_ocm_masks': use_ocm_masks, 'color_jitter': color_jitter,
        'channel_kinds': channel_kinds, 'stretch': stretch, 'folds': folds_csv,
        'fold': fold, 'train_split': list(train_split) if isinstance(train_split, tuple)
        else [train_split], 'test_split': list(test_split) if isinstance(test_split, tuple)
        else [test_split], 'train_chips': train_chips, 'test_chips': test_chips,
        'seed': seed,
    }

    device = select_device()
    common = dict(size=size, min_instance_size=min_instance_size,
                  use_ocm_masks=use_ocm_masks, channel_kinds=channel_kinds,
                  stretch=stretch)
    train_ds = SemanticWindows(PlanetMaskRCNNDataset(
        imagedir, split=train_split, color_jitter=color_jitter,
        chip_names=train_chips, **common), use_ocm_masks)
    # Worker processes only on CUDA. On macOS torch shares tensors through the
    # file_system strategy, whose per-worker torch_shm_manager daemons inherit the
    # multiprocessing resource tracker's pipe and deadlock interpreter exit.
    num_workers = 2 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    test_loader = None
    if test_chips is None or len(test_chips):
        test_ds = SemanticWindows(PlanetMaskRCNNDataset(
            imagedir, split=test_split, color_jitter=False, chip_names=test_chips,
            **common), use_ocm_masks)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)
    print(f'{len(train_ds)} train windows, '
          f'{len(test_loader.dataset) if test_loader else 0} test windows, device {device}')

    model = DinoSegmenter(backbone, finetune_blocks, decoder_dim,
                          image_stem=image_stem).to(device)
    model.min_pred_size = min_pred_size
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    groups = [{'params': head, 'lr': lr}] + ([{'params': bb, 'lr': backbone_lr}] if bb else [])
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    n_train = sum(p.numel() for p in head + bb)
    print(f'{backbone}: {n_train / 1e6:.1f} M trainable parameters')
    autocast = (torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda'
                else torch.autocast('cpu', enabled=False))

    metrics_path = os.path.join(outputdir, 'metrics.jsonl')
    open(metrics_path, 'w').close()
    for epoch in range(1, num_epochs + 1):
        model.train()
        model.backbone.eval()  # no dropout or drop-path in the backbone
        t0, total, n = time.time(), 0.0, 0
        for img, target, ignore in train_loader:
            img, target, ignore = img.to(device), target.to(device), ignore.to(device)
            with autocast:
                logits = model(img)
            loss = bce_dice_loss(logits.float(), target, ignore)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * img.shape[0]
            n += img.shape[0]
        row = {'epoch': epoch, 'train_loss': total / n, 'seconds': time.time() - t0}
        if test_loader is not None:
            model.eval()
            with autocast:
                row['test_iou'] = window_iou(model, test_loader, device)
        print(json.dumps(row))
        with open(metrics_path, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        ckpt = {'kind': 'dinov3', 'epoch': epoch, 'params': params,
                'state_dict': model.trainable_state_dict()}
        if epoch in save_epochs:
            torch.save(ckpt, os.path.join(outputdir, f'epoch_{epoch:03d}.pth'))
        if epoch == num_epochs:
            torch.save(ckpt, os.path.join(outputdir, 'last_model.pth'))


if __name__ == '__main__':
    main()
