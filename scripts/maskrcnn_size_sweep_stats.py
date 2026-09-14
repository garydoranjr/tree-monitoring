#!/usr/bin/env python
"""Cache everything needed to sweep a post-hoc crown-size threshold.

Runs one Mask R-CNN checkpoint over the test split and writes, per chip, the
predicted instance areas and scores, the ground-truth instance areas, and the
full (P, G) instance-mask IoU matrix. Those arrays fully determine TP/FP/FN at
*any* post-hoc minimum-area threshold, so the threshold sweep itself
(`plot_maskrcnn_size_sweep.py`) needs no GPU and no re-inference.

Predictions are produced exactly as the training-loop evaluator does
(`train_planet_image_maskrcnn.evaluate`), including the OCM cloud filter that
drops detections whose box center falls on a cloudy pixel. The unfiltered
TP/FP/FN are also stored as `baseline_*` so the cache can be checked against
the `test_event/*` values wandb logged for the same checkpoint.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede torch

import re
import __main__

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_planet_image_maskrcnn as T
from train_planet_image_maskrcnn import (
    PlanetMaskRCNNDataset,
    _greedy_match,
    _lookup_clear_at_centers,
    collate_fn,
    iou_matrix_torch,
)
from util import select_device

EPOCH_PATTERN = re.compile(r'epoch_(\d+)\.pth$')


def load_checkpoint(modelfile):
    """Unpickle a training checkpoint, returning (model, params).

    The checkpoints hold the whole pickled model object, saved from a script
    run as ``__main__``, so every custom class is recorded as ``__main__.X``.
    Rebinding them onto the real ``__main__`` is what lets the unpickler
    resolve ``OCMRPN`` and ``OCMRoIHeads``, which are referenced by the
    pickled OCM model but are not otherwise present here."""
    for name in ('OCMMaskRCNN', 'OCMRPN', 'OCMRoIHeads'):
        setattr(__main__, name, getattr(T, name))
    ckpt = torch.load(modelfile, map_location='cpu', weights_only=False)
    return ckpt['model'], ckpt['params']


@torch.no_grad()
def collect_stats(model, dataloader, device, mask_thresh, iou_thresh,
                  use_ocm_masks):
    """Run the model over the loader, returning per-chip areas/scores/IoU.

    Mirrors `train_planet_image_maskrcnn.evaluate` so the cached numbers are
    directly comparable to the metrics logged during training."""
    model.eval()

    pred_areas, pred_scores, gt_areas, ious = [], [], [], []
    base_tp = base_fp = base_fn = 0

    for imgs, targets in tqdm(dataloader, desc='chips'):
        img = imgs[0].to(device)
        tgt = targets[0]
        out = model([img])[0]

        h, w = img.shape[-2:]
        masks = out['masks']
        scores = out['scores']
        boxes = out['boxes']
        if masks.numel() == 0:
            bin_masks = torch.zeros((0, h, w), dtype=torch.bool, device=device)
        else:
            bin_masks = (masks[:, 0] >= mask_thresh)

        if use_ocm_masks and boxes.shape[0] > 0:
            clear = tgt['clear_mask'].to(device).bool()
            keep = _lookup_clear_at_centers([clear], boxes, 0)
            bin_masks = bin_masks[keep]
            scores = scores[keep]

        gt_masks = tgt['masks'].to(device).bool()

        iou = iou_matrix_torch(bin_masks, gt_masks).cpu().numpy()
        scores_np = scores.detach().cpu().numpy().astype(np.float32)

        # Zero-area predictions (a detection whose mask vanishes at
        # mask_thresh) are kept so baseline_* reproduces evaluate() exactly;
        # any post-hoc threshold >= 1 discards them anyway.
        # Summing over the spatial dims (rather than reshaping to 2D) keeps
        # this valid for chips where nothing was detected at all.
        pred_areas.append(
            bin_masks.sum(dim=(1, 2)).cpu().numpy().astype(np.int32)
        )
        gt_areas.append(
            gt_masks.sum(dim=(1, 2)).cpu().numpy().astype(np.int32)
        )
        pred_scores.append(scores_np)
        ious.append(iou.astype(np.float32))

        cls = _greedy_match(iou, scores_np, iou_thresh=iou_thresh)
        base_tp += cls['pred_labels'].count('TP')
        base_fp += cls['pred_labels'].count('FP')
        base_fn += cls['gt_labels'].count('FN')

    n_pred = np.array([a.size for a in pred_areas], dtype=np.int32)
    n_gt = np.array([a.size for a in gt_areas], dtype=np.int32)
    iou_offset = np.concatenate(
        [[0], np.cumsum([m.size for m in ious])]
    ).astype(np.int64)

    return {
        'pred_area': np.concatenate(pred_areas) if pred_areas else np.zeros(0, np.int32),
        'pred_score': np.concatenate(pred_scores) if pred_scores else np.zeros(0, np.float32),
        'gt_area': np.concatenate(gt_areas) if gt_areas else np.zeros(0, np.int32),
        'n_pred': n_pred,
        'n_gt': n_gt,
        'iou_flat': (
            np.concatenate([m.ravel() for m in ious])
            if ious else np.zeros(0, np.float32)
        ),
        'iou_offset': iou_offset,
        'baseline_tp': base_tp,
        'baseline_fp': base_fp,
        'baseline_fn': base_fn,
    }


@click.command()
@click.argument('modelfile')
@click.argument('imagedir')
@click.argument('outputfile')
@click.option('--split', default='right',
              help='Chip split to evaluate; training used the right half.')
@click.option('--size', default=None, type=int,
              help='Crop size; defaults to the checkpoint value.')
@click.option('--mask-thresh', default=0.5, type=float,
              help='Mask-probability threshold used to binarize predicted '
                   'masks. This is NOT the detector confidence threshold, '
                   'which is baked into the checkpoint (0.05).')
@click.option('--iou-thresh', default=0.5, type=float,
              help='IoU required to call a prediction a true positive.')
def main(modelfile, imagedir, outputfile, split, size, mask_thresh,
         iou_thresh):
    model, params = load_checkpoint(modelfile)
    device = select_device()
    model.to(device)

    size = size or params['size']
    min_instance_size = params['min_instance_size']
    use_ocm_masks = params['use_ocm_masks']
    m = EPOCH_PATTERN.search(os.path.basename(modelfile))
    epoch = int(m.group(1)) if m else -1

    print(
        f"{os.path.basename(modelfile)}: min_instance_size={min_instance_size} "
        f"epoch={epoch} channels={params['channel_kinds']} "
        f"ocm={use_ocm_masks} device={device}"
    )

    dataset = PlanetMaskRCNNDataset(
        imagedir, split=split, size=size, color_jitter=False,
        min_instance_size=min_instance_size, use_ocm_masks=use_ocm_masks,
        channel_kinds=params['channel_kinds'],
    )
    print(f"Evaluating {len(dataset)} chips from {imagedir} (split={split})")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
    )

    stats = collect_stats(
        model, loader, device, mask_thresh, iou_thresh, use_ocm_masks,
    )

    # The training filter drops instances with area <= min_instance_size, so
    # no cached GT crown may sit at or below it. This is what makes a post-hoc
    # threshold equal to the training threshold a guaranteed no-op.
    assert (stats['gt_area'] > min_instance_size).all(), (
        "cached GT contains instances at or below the training threshold"
    )

    tp, fp, fn = stats['baseline_tp'], stats['baseline_fp'], stats['baseline_fn']
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    print(
        f"baseline (no post-hoc filter): tp={tp} fp={fp} fn={fn} "
        f"precision={precision:.4f} recall={recall:.4f}"
    )

    np.savez_compressed(
        outputfile,
        chip_names=np.array(
            [os.path.basename(f) for f in dataset.mask_files], dtype='U'
        ),
        min_instance_size=min_instance_size,
        epoch=epoch,
        mask_thresh=mask_thresh,
        iou_thresh=iou_thresh,
        split=split,
        size=size,
        use_ocm_masks=use_ocm_masks,
        **stats,
    )
    print(f"Wrote {outputfile}")


if __name__ == '__main__':
    main()
