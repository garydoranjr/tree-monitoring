#!/usr/bin/env python
import os
import glob
import warnings
import wandb
import click
import rasterio
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.ops import masks_to_boxes
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision


def _split_window(h, w, split, size):
    if h < size:
        raise ValueError(f"Height must be at least {size}, got {h}")
    if w < 2 * size:
        raise ValueError(f"Width must be at least {2*size}, got {w}")
    if split not in ("left", "right"):
        raise ValueError('split must be either "left" or "right"')

    center_row = h // 2
    row_start = center_row - size // 2
    row_end = row_start + size
    if row_start < 0:
        row_start = 0
        row_end = size
    if row_end > h:
        row_end = h
        row_start = h - size

    center_col = w // 2
    if split == "left":
        col_end = center_col
        col_start = col_end - size
    else:
        col_start = center_col
        col_end = col_start + size

    if col_start < 0 or col_end > w:
        raise ValueError("Computed window exceeds image bounds")

    return row_start, row_end, col_start, col_end


def get_split(img, mask, split, size, *extra):
    img = np.asarray(img)
    mask = np.asarray(mask)

    if img.ndim < 2 or mask.ndim < 2:
        raise ValueError("img and mask must have at least 2 dimensions")
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("img and mask must match in their first two dimensions")
    extra = [np.asarray(a) for a in extra]
    for a in extra:
        if a.shape[:2] != img.shape[:2]:
            raise ValueError("extra arrays must match img in first two dims")

    h, w = img.shape[:2]
    row_start, row_end, col_start, col_end = _split_window(h, w, split, size)

    img_crop = img[row_start:row_end, col_start:col_end, ...]
    mask_crop = mask[row_start:row_end, col_start:col_end, ...]
    if not extra:
        return img_crop, mask_crop
    extra_crops = [a[row_start:row_end, col_start:col_end, ...] for a in extra]
    return (img_crop, mask_crop, *extra_crops)


# Channel "kind" -> source band index in a 4-band tif read as (H, W, 4),
# whose band order is (Blue, Green, Red, NIR) per apply_drone_labels_coreg.py.
_KIND_SOURCE_IDX = {'blue': 0, 'green': 1, 'red': 2, 'ir': 3}

# Per-kind ImageNet-style normalization (mean, std). The color kinds use the
# standard torchvision stats; 'ir' reuses the red-channel stats. 'ndvi' is
# normalized with dataset-derived stats computed at training time, so it is
# absent here.
_KIND_NORM = {
    'red': (0.485, 0.229),
    'green': (0.456, 0.224),
    'blue': (0.406, 0.225),
    'ir': (0.485, 0.229),
}


def resolve_channels(fourth_band, replace):
    """Resolve the (--fourth-band, --replace) flags into an ordered list of
    per-channel "kinds" describing the model input, built from the 4-band tif.

    The three color positions are in true RGB order so the input aligns with
    the pretrained conv1. With ``replace == 'none'`` the fourth band (NIR or
    NDVI) is appended as a real 4th channel; otherwise it substitutes the
    named color channel and the input stays 3-channel.
    """
    if fourth_band == 'none':
        if replace != 'none':
            raise ValueError(
                "--replace requires --fourth-band to be 'ir' or 'ndvi'; there "
                "is nothing to substitute when --fourth-band is 'none'."
            )
        return ['red', 'green', 'blue']
    fourth_kind = 'ir' if fourth_band == 'ir' else 'ndvi'
    kinds = ['red', 'green', 'blue']
    if replace == 'none':
        kinds.append(fourth_kind)
    else:
        kinds[{'r': 0, 'g': 1, 'b': 2}[replace]] = fourth_kind
    return kinds


def _compute_ndvi(arr):
    """NDVI = (NIR - Red) / (NIR + Red) from a (H, W, 4) (B, G, R, NIR) array,
    with a zero-denominator guard. Returns raw values in [-1, 1]."""
    red = arr[..., _KIND_SOURCE_IDX['red']]
    nir = arr[..., _KIND_SOURCE_IDX['ir']]
    denom = nir + red
    denom = np.where(denom == 0, 1e-9, denom)
    return (nir - red) / denom


def build_input_channels(arr, channel_kinds):
    """Assemble the (H, W, C) model-input array from a (H, W, 4) (Blue, Green,
    Red, NIR) float array per `channel_kinds`. Color/IR channels get a per-band
    0-99.9 percentile stretch to [0, 1]; the NDVI channel is left raw in
    [-1, 1] (it is normalized downstream with dataset stats). Shared by the
    training dataset and the deploy scripts so the input recipe stays in sync."""
    ndvi = _compute_ndvi(arr) if 'ndvi' in channel_kinds else None
    channels = []
    for kind in channel_kinds:
        if kind == 'ndvi':
            channels.append(ndvi)
        else:
            band = arr[..., _KIND_SOURCE_IDX[kind]]
            p_low, p_high = np.percentile(band, (0, 99.9))
            band = np.clip((band - p_low) / (p_high - p_low + 1e-8), 0, 1)
            channels.append(band)
    return np.stack(channels, axis=-1).astype(np.float32)


def compute_ndvi_stats(img_files, split, size):
    """Mean/std of raw NDVI over the train-split crop of every chip, used to
    normalize the NDVI channel. Single pass; returns (mean, std)."""
    total = 0.0
    total_sq = 0.0
    count = 0
    for path in img_files:
        with rasterio.open(path) as src:
            data = src.read()
        arr = data.transpose(1, 2, 0).astype(np.float32)
        ndvi = _compute_ndvi(arr)
        h, w = ndvi.shape[:2]
        r0, r1, c0, c1 = _split_window(h, w, split, size)
        crop = ndvi[r0:r1, c0:c1]
        total += float(crop.sum())
        total_sq += float(np.square(crop).sum())
        count += crop.size
    if count == 0:
        raise ValueError("No chips found to compute NDVI statistics")
    mean = total / count
    var = max(total_sq / count - mean * mean, 1e-12)
    return mean, float(np.sqrt(var))


def binary_mask_to_instances(bin_mask, min_instance_size=4):
    """Convert a 2D binary mask to a (N, H, W) uint8 stack of per-instance
    masks. Drops instances smaller than min_instance_size pixels and any
    instance that touches the crop border."""
    h, w = bin_mask.shape
    labeled, n = ndimage.label(bin_mask, structure=np.ones((3, 3), dtype=np.uint8))
    instances = []
    for i in range(1, n + 1):
        inst = labeled == i
        if inst.sum() <= min_instance_size:
            continue
        rows = np.any(inst, axis=1)
        cols = np.any(inst, axis=0)
        if rows[0] or rows[-1] or cols[0] or cols[-1]:
            continue
        instances.append(inst.astype(np.uint8))
    if not instances:
        return np.zeros((0, h, w), dtype=np.uint8)
    return np.stack(instances, axis=0)


def _greedy_match(iou, pred_scores, iou_thresh=0.5):
    """Greedy TP/FP/FN labeling given a (P, G) IoU matrix and predicted
    confidence scores. Inputs are NumPy; the loop is small (P*G entries)
    so CPU is fine even when the IoU itself was built on GPU."""
    P, G = iou.shape
    pred_labels = ['FP'] * P
    gt_labels = ['FN'] * G
    pred_to_gt = [-1] * P
    if P and G:
        order = np.argsort(-pred_scores)
        gt_used = np.zeros(G, dtype=bool)
        for pi in order:
            if gt_used.all():
                break
            avail = np.where(~gt_used)[0]
            best_local = int(np.argmax(iou[pi, avail]))
            gi = int(avail[best_local])
            if iou[pi, gi] >= iou_thresh:
                gt_used[gi] = True
                pred_labels[pi] = 'TP'
                gt_labels[gi] = 'TP'
                pred_to_gt[pi] = gi
    return {
        'pred_labels': pred_labels,
        'gt_labels': gt_labels,
        'pred_to_gt': pred_to_gt,
        'iou': iou,
    }


def iou_matrix_torch(pred_masks, gt_masks):
    """Compute the (P, G) instance-mask IoU matrix on whatever device the
    inputs live on. Inputs are (P, H, W) and (G, H, W) tensors (bool or
    numeric). Returns a (P, G) float32 tensor on the same device.

    Casting to float32 lets the matmul go through BLAS/cuBLAS; integer
    matmul has no fast path in either NumPy or torch."""
    P = pred_masks.shape[0]
    G = gt_masks.shape[0]
    device = pred_masks.device
    if P == 0 or G == 0:
        return torch.zeros((P, G), dtype=torch.float32, device=device)
    p_flat = pred_masks.reshape(P, -1).to(torch.float32)
    g_flat = gt_masks.reshape(G, -1).to(torch.float32)
    inter = p_flat @ g_flat.t()
    p_area = p_flat.sum(dim=1, keepdim=True)
    g_area = g_flat.sum(dim=1, keepdim=True).t()
    union = p_area + g_area - inter
    iou = torch.where(union > 0, inter / union.clamp(min=1.0),
                      torch.zeros_like(inter))
    return iou


def classify_instances_torch(gt_masks, pred_masks, pred_scores,
                             iou_thresh=0.5):
    """Torch-tensor entry point used by the training-loop evaluator.

    `gt_masks` and `pred_masks` are (G, H, W) and (P, H, W) tensors on
    the model's device; `pred_scores` is a (P,) tensor. Only the small
    (P, G) IoU matrix and (P,) scores cross the GPU->CPU boundary."""
    iou_t = iou_matrix_torch(pred_masks, gt_masks)
    iou = iou_t.detach().cpu().numpy().astype(np.float32)
    scores = pred_scores.detach().cpu().numpy()
    return _greedy_match(iou, scores, iou_thresh=iou_thresh)


def classify_instances(gt_masks, pred_masks, pred_scores, iou_thresh=0.5):
    """Greedy IoU matching between predicted and GT instance masks.

    NumPy entry point used by the interactive viewer. Returns per-instance
    TP/FP/FN labels plus the raw IoU matrix. The UI filter hook for
    TP/FP/FN lives off of this dict."""
    P = pred_masks.shape[0]
    G = gt_masks.shape[0]
    iou = np.zeros((P, G), dtype=np.float32)
    if P and G:
        p_flat = pred_masks.reshape(P, -1).astype(np.float32)
        g_flat = gt_masks.reshape(G, -1).astype(np.float32)
        inter = p_flat @ g_flat.T
        p_area = p_flat.sum(axis=1)[:, None]
        g_area = g_flat.sum(axis=1)[None, :]
        union = p_area + g_area - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return _greedy_match(iou, pred_scores, iou_thresh=iou_thresh)


_warned_missing_ocm = False


def _load_clear_mask(img_path, expected_shape):
    """Load the OCM sidecar `{stem}.ocm.png` and return a bool clear-mask of
    `expected_shape` (H, W). The sidecar is RGBA with alpha=150 over cloudy
    pixels and alpha=0 elsewhere, so `clear = (alpha == 0)`. Missing files
    fall back to all-clear with a one-shot warning."""
    global _warned_missing_ocm
    ocm_path = os.path.splitext(img_path)[0] + '.ocm.png'
    if not os.path.exists(ocm_path):
        if not _warned_missing_ocm:
            warnings.warn(
                f"OCM sidecar not found (e.g. {ocm_path}); treating affected "
                "samples as fully clear. This warning is shown once.",
                stacklevel=2,
            )
            _warned_missing_ocm = True
        return np.ones(expected_shape, dtype=bool)
    arr = np.array(Image.open(ocm_path).convert('RGBA'))
    alpha = arr[..., 3]
    if alpha.shape != expected_shape:
        raise ValueError(
            f"OCM sidecar {ocm_path} shape {alpha.shape} does not match "
            f"image shape {expected_shape}"
        )
    return (alpha == 0)


class PlanetMaskRCNNDataset(Dataset):

    def __init__(self, img_dir, split, size=512, color_jitter=False,
                 min_instance_size=4, use_ocm_masks=False,
                 channel_kinds=('red', 'green', 'blue')):
        self.channel_kinds = list(channel_kinds)
        self.mask_files = sorted(glob.glob(os.path.join(img_dir, "*.mask.png")))
        # All chips are uint16 GeoTIFFs with band order (Blue, Green, Red, NIR)
        # produced by apply_drone_labels_coreg.py; the model input channels are
        # assembled from them per `channel_kinds`.
        self.img_files = [
            mf.replace('.mask.png', '.tif') for mf in self.mask_files
        ]
        self.split = split
        self.size = size
        self.min_instance_size = min_instance_size
        self.use_ocm_masks = use_ocm_masks
        self.jitter = (
            T.ColorJitter(brightness=0.2, contrast=0.2) if color_jitter else None
        )

    def __len__(self):
        return len(self.img_files)

    def _empty_target(self, idx, clear_crop):
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, self.size, self.size), dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }
        if self.use_ocm_masks:
            target["clear_mask"] = torch.from_numpy(
                clear_crop.astype(np.uint8)
            )
        return target

    def _jitter_one(self, ch):
        """Jitter a single [0,1] channel by replicating it to 3 channels,
        applying ColorJitter, and taking the first channel back (ColorJitter
        only accepts 3-channel images)."""
        g = (ch * 255).astype(np.uint8)
        g3 = np.stack([g, g, g], axis=-1)
        out = np.array(self.jitter(Image.fromarray(g3)))[..., 0]
        return out.astype(np.float32) / 255.0

    def _apply_jitter(self, img_crop):
        """ColorJitter the stretched ([0,1]) channels. The three color/IR
        channels are jittered jointly as an RGB image when none of the first
        three is NDVI; otherwise each stretched channel is jittered on its
        own. An appended IR channel (position 3) is jittered as a single
        channel. The raw NDVI channel is left untouched (it is outside the
        uint8 ColorJitter domain)."""
        img_crop = img_crop.copy()
        kinds = self.channel_kinds
        if 'ndvi' not in kinds[:3]:
            rgb = (img_crop[..., :3] * 255).astype(np.uint8)
            img_crop[..., :3] = (
                np.array(self.jitter(Image.fromarray(rgb))).astype(np.float32)
                / 255.0
            )
        else:
            for i in range(3):
                if kinds[i] != 'ndvi':
                    img_crop[..., i] = self._jitter_one(img_crop[..., i])
        if len(kinds) == 4 and kinds[3] != 'ndvi':
            img_crop[..., 3] = self._jitter_one(img_crop[..., 3])
        return img_crop

    def __getitem__(self, idx):
        # Read the 4-band tif (Blue, Green, Red, NIR) and assemble the model
        # input channels per self.channel_kinds. Color/IR channels get a
        # per-band 0-99.9 percentile stretch to [0, 1]; the NDVI channel is
        # left raw in [-1, 1] (it is normalized later with dataset stats).
        with rasterio.open(self.img_files[idx]) as src:
            data = src.read()  # (4, H, W) uint16
        arr = data.transpose(1, 2, 0).astype(np.float32)
        img = build_input_channels(arr, self.channel_kinds)  # (H, W, C)
        mask = np.array(Image.open(self.mask_files[idx]))
        mask = (mask == 255).astype(np.uint8)

        if self.use_ocm_masks:
            clear = _load_clear_mask(self.img_files[idx], mask.shape)
            img_crop, mask_crop, clear_crop = get_split(
                img, mask, self.split, self.size, clear,
            )
        else:
            img_crop, mask_crop = get_split(img, mask, self.split, self.size)
            clear_crop = None

        inst_masks = binary_mask_to_instances(
            mask_crop, min_instance_size=self.min_instance_size,
        )

        if self.jitter is not None:
            img_crop = self._apply_jitter(img_crop)
        img_tensor = torch.from_numpy(
            img_crop.transpose(2, 0, 1).copy()
        ).float()  # (C, H, W) float32

        if inst_masks.shape[0] == 0:
            return img_tensor, self._empty_target(idx, clear_crop)

        masks_t = torch.from_numpy(inst_masks)  # (N, H, W) uint8
        boxes = masks_to_boxes(masks_t)         # (N, 4) float, inclusive xmax/ymax
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        masks_t = masks_t[keep]
        boxes = boxes[keep]
        if masks_t.shape[0] == 0:
            return img_tensor, self._empty_target(idx, clear_crop)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        labels = torch.ones((masks_t.shape[0],), dtype=torch.int64)
        target = {
            "boxes": boxes.to(torch.float32),
            "labels": labels,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": areas.to(torch.float32),
            "iscrowd": torch.zeros((masks_t.shape[0],), dtype=torch.int64),
        }
        if self.use_ocm_masks:
            target["clear_mask"] = torch.from_numpy(
                clear_crop.astype(np.uint8)
            )
        return img_tensor, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def _lookup_clear_at_centers(clear_masks_per_image, boxes_per_image,
                             image_index):
    """Sample the clear-mask at each box's center pixel. `clear_masks_per_image`
    is a list of (H, W) bool tensors aligned with the post-transform image
    coordinates; `boxes_per_image` is an (N, 4) tensor of [x1, y1, x2, y2].
    Out-of-bounds centers (in padding regions) are treated as cloudy
    (returned False) so the corresponding anchors/proposals are ignored."""
    cm = clear_masks_per_image[image_index]
    H, W = cm.shape
    if boxes_per_image.numel() == 0:
        return torch.ones((0,), dtype=torch.bool, device=cm.device)
    cx = ((boxes_per_image[:, 0] + boxes_per_image[:, 2]) * 0.5).long()
    cy = ((boxes_per_image[:, 1] + boxes_per_image[:, 3]) * 0.5).long()
    in_bounds = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    cx = cx.clamp(0, W - 1)
    cy = cy.clamp(0, H - 1)
    sampled = cm[cy, cx]
    return sampled & in_bounds


class OCMRPN(RegionProposalNetwork):
    """RPN that marks anchors centered on cloudy pixels as ignore (label=-1)
    so they're skipped by `BalancedPositiveNegativeSampler` and contribute
    to neither `loss_objectness` nor `loss_rpn_box_reg`. The cloudy mask is
    read from `self._clear_masks`, populated by `OCMMaskRCNN.forward`."""

    def assign_targets_to_anchors(self, anchors, targets):
        labels, matched_gt_boxes = super().assign_targets_to_anchors(
            anchors, targets,
        )
        clear_masks = getattr(self, '_clear_masks', None)
        if clear_masks is None:
            return labels, matched_gt_boxes
        for i, anchors_i in enumerate(anchors):
            keep_clear = _lookup_clear_at_centers(clear_masks, anchors_i, i)
            labels[i] = torch.where(
                keep_clear, labels[i], torch.full_like(labels[i], -1.0),
            )
        return labels, matched_gt_boxes


class OCMRoIHeads(RoIHeads):
    """RoI heads that mark proposals centered on cloudy pixels as ignore
    (label=-1) before subsampling, so they contribute to none of
    `loss_classifier`, `loss_box_reg`, or `loss_mask`."""

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs, labels = super().assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels,
        )
        clear_masks = getattr(self, '_clear_masks', None)
        if clear_masks is None:
            return matched_idxs, labels
        for i, proposals_i in enumerate(proposals):
            keep_clear = _lookup_clear_at_centers(clear_masks, proposals_i, i)
            labels[i] = torch.where(
                keep_clear, labels[i], torch.full_like(labels[i], -1),
            )
        return matched_idxs, labels


class OCMMaskRCNN(MaskRCNN):
    """Mask R-CNN wrapper that pulls per-image `clear_mask` tensors out of
    the targets, resizes them to match the post-transform image size, and
    stashes them on the RPN and RoI heads so anchors/proposals over cloudy
    pixels are ignored in every loss term."""

    def forward(self, images, targets=None):
        clear_masks = None
        if self.training and targets is not None:
            popped = []
            any_present = False
            for t in targets:
                cm = t.pop('clear_mask', None)
                popped.append(cm)
                if cm is not None:
                    any_present = True
            if any_present:
                clear_masks = popped

        if clear_masks is None or not self.training:
            return super().forward(images, targets)

        original_image_sizes = [tuple(img.shape[-2:]) for img in images]
        images_t, targets_t = self.transform(images, targets)

        post_h, post_w = images_t.tensors.shape[-2:]
        cm_resized = []
        for cm, orig_hw, post_hw in zip(
            clear_masks, original_image_sizes, images_t.image_sizes,
        ):
            if cm is None:
                cm_t = torch.ones(
                    orig_hw, dtype=torch.bool, device=images_t.tensors.device,
                )
            else:
                cm_t = cm.to(
                    device=images_t.tensors.device, dtype=torch.bool,
                )
                if cm_t.shape != orig_hw:
                    raise ValueError(
                        f"clear_mask shape {tuple(cm_t.shape)} does not match "
                        f"image shape {orig_hw}"
                    )
            cm_resized_unpadded = F.interpolate(
                cm_t[None, None].float(), size=post_hw, mode='nearest',
            )[0, 0].bool()
            padded = torch.zeros(
                (post_h, post_w), dtype=torch.bool,
                device=images_t.tensors.device,
            )
            ph, pw = post_hw
            padded[:ph, :pw] = cm_resized_unpadded
            cm_resized.append(padded)

        if targets_t is not None:
            for target_idx, target in enumerate(targets_t):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and "
                        f"width. Found invalid box {degen_bb} for target at "
                        f"index {target_idx}."
                    )

        from collections import OrderedDict
        features = self.backbone(images_t.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        self.rpn._clear_masks = cm_resized
        self.roi_heads._clear_masks = cm_resized
        try:
            proposals, proposal_losses = self.rpn(images_t, features, targets_t)
            detections, detector_losses = self.roi_heads(
                features, proposals, images_t.image_sizes, targets_t,
            )
        finally:
            self.rpn._clear_masks = None
            self.roi_heads._clear_masks = None

        detections = self.transform.postprocess(
            detections, images_t.image_sizes, original_image_sizes,
        )
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return self.eager_outputs(losses, detections)


def build_model(num_classes=2, nms_thresh=0.3, score_thresh=0.05,
                detections_per_img=300, use_ocm_masks=False,
                image_mean=None, image_std=None, nir_init='zero'):
    if image_mean is None or image_std is None:
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
    n_input_channels = len(image_mean)
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    if n_input_channels == 4:
        # Replace the backbone's first conv from 3->4 input channels. Copy the
        # pretrained COCO weights into the RGB channels (the input is in true
        # RGB order, matching the pretrained conv1). The appended 4th channel
        # (NIR or NDVI) is initialized per `nir_init`: 'zero' zero-inits it
        # (the model starts equivalent to the pretrained 3-band model and
        # learns the 4th-band contribution from scratch), while
        # 'red'/'green'/'blue' duplicate that pretrained channel's filter into
        # the new channel to give it a sensible non-zero starting response.
        old_conv = model.backbone.body.conv1
        new_conv = torch.nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        _RGB_IDX = {'red': 0, 'green': 1, 'blue': 2}
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if nir_init == 'zero':
                new_conv.weight[:, 3:4, :, :] = 0.0
            else:
                src = _RGB_IDX[nir_init]
                new_conv.weight[:, 3:4, :, :] = old_conv.weight[:, src:src + 1, :, :]
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        model.backbone.body.conv1 = new_conv

    # The detection transform normalizes per channel and needs one value per
    # input band; set both 3- and 4-channel cases from the resolved stats.
    model.transform.image_mean = list(image_mean)
    model.transform.image_std = list(image_std)

    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=aspect_ratios,
    )
    model.rpn.anchor_generator = anchor_generator
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    in_channels = model.rpn.head.conv[0][0].in_channels
    from torchvision.models.detection.rpn import RPNHead
    model.rpn.head = RPNHead(in_channels, num_anchors)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes,
    )

    model.roi_heads.nms_thresh = nms_thresh
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.detections_per_img = detections_per_img

    if use_ocm_masks:
        model.__class__ = OCMMaskRCNN
        model.rpn.__class__ = OCMRPN
        model.roi_heads.__class__ = OCMRoIHeads
        model.rpn._clear_masks = None
        model.roi_heads._clear_masks = None
    return model


@torch.no_grad()
def evaluate(model, dataloader, device, iou_metric, map_metric,
             score_thresh=0.5, iou_thresh=0.5, use_ocm_masks=False):
    """Compute binary-IoU (union of masks), mask mAP, and event-level
    precision/recall/accuracy on the dataloader. When `use_ocm_masks` is
    True, predictions whose box centers fall in cloudy pixels are dropped
    before metrics are accumulated, and the binary-IoU metric is restricted
    to clear pixels."""
    model.eval()
    iou_metric.reset()
    map_metric.reset()

    tp = 0
    fp = 0
    fn = 0

    for imgs, targets in dataloader:
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)

        clear_masks = None
        if use_ocm_masks:
            clear_masks = [
                tgt['clear_mask'].to(device).bool() for tgt in targets
            ]

        preds_for_map = []
        for i, out in enumerate(outputs):
            masks = out["masks"]  # (N, 1, H, W) float
            scores = out["scores"]
            labels = out["labels"]
            boxes = out["boxes"]
            if masks.numel() == 0:
                bin_masks = torch.zeros(
                    (0, imgs[0].shape[-2], imgs[0].shape[-1]),
                    dtype=torch.bool, device=device,
                )
            else:
                bin_masks = (masks[:, 0] >= score_thresh)

            if clear_masks is not None and boxes.shape[0] > 0:
                keep = _lookup_clear_at_centers(clear_masks, boxes, i)
                bin_masks = bin_masks[keep]
                scores = scores[keep]
                labels = labels[keep]
                boxes = boxes[keep]

            preds_for_map.append({
                "masks": bin_masks,
                "scores": scores,
                "labels": labels,
                "boxes": boxes,
            })

        gts_for_map = []
        for tgt in targets:
            gts_for_map.append({
                "masks": tgt["masks"].to(device).bool(),
                "labels": tgt["labels"].to(device),
                "boxes": tgt["boxes"].to(device),
            })

        for i, (pred, tgt) in enumerate(zip(preds_for_map, gts_for_map)):
            h, w = imgs[0].shape[-2:]
            pred_union = (
                pred["masks"].any(dim=0)
                if pred["masks"].shape[0] > 0
                else torch.zeros((h, w), dtype=torch.bool, device=device)
            )
            gt_union = (
                tgt["masks"].any(dim=0)
                if tgt["masks"].shape[0] > 0
                else torch.zeros((h, w), dtype=torch.bool, device=device)
            )
            if clear_masks is not None:
                cm = clear_masks[i]
                pred_union = pred_union & cm
                gt_union = gt_union & cm
            iou_metric.update(pred_union.int(), gt_union.int())

            cls = classify_instances_torch(
                tgt["masks"], pred["masks"], pred["scores"],
                iou_thresh=iou_thresh,
            )
            tp += cls['pred_labels'].count('TP')
            fp += cls['pred_labels'].count('FP')
            fn += cls['gt_labels'].count('FN')

        map_metric.update(preds_for_map, gts_for_map)

    iou = iou_metric.compute().item()
    map_results = map_metric.compute()

    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    accuracy = (
        tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
    )
    event_metrics = {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }
    return iou, map_results, event_metrics


@click.command()
@click.argument('imagedir')
@click.argument('outputdir')
@click.option('--num-epochs', default=200, type=int)
@click.option('--batch-size', default=4, type=int)
@click.option('--lr', default=1e-4, type=float)
@click.option('--size', default=512, type=int)
@click.option('--min-instance-size', default=4, type=int)
@click.option('--nms-thresh', default=0.3, type=float)
@click.option('--score-thresh', default=0.05, type=float)
@click.option('--detections-per-img', default=300, type=int)
@click.option('--ocm-masks/--no-ocm-masks', 'use_ocm_masks', default=False,
              help='Use OCM cloud-mask sidecars (.ocm.png) to ignore '
                   'anchors/proposals/predictions over cloudy pixels in '
                   'every loss term and in crown-level TP/FP/FN.')
@click.option('--fourth-band', default='none',
              type=click.Choice(['none', 'ir', 'ndvi']),
              help='Extra channel derived from the 4-band GeoTIFF chips: '
                   "'none' = RGB only (default), 'ir' = raw NIR, "
                   "'ndvi' = NDVI computed from Red and NIR.")
@click.option('--replace', default='none',
              type=click.Choice(['none', 'r', 'g', 'b']),
              help="What to do with the --fourth-band channel: 'none' "
                   '(default) appends it as a real 4th channel (4-channel '
                   "model); 'r'/'g'/'b' substitutes it in place of that color "
                   'channel (stays a 3-channel model).')
@click.option('--nir-init', default='zero',
              type=click.Choice(['zero', 'red', 'green', 'blue']),
              help='Appended-4th-channel only: how to initialize the new input '
                   "channel (NIR or NDVI) of the first conv. 'zero' (default) "
                   "zero-inits it; 'red'/'green'/'blue' duplicate that "
                   "pretrained channel's weights into the new filter.")
@click.option('--wandb/--no-wandb', 'use_wandb', default=True)
def main(imagedir, outputdir, num_epochs, batch_size, lr, size,
         min_instance_size, nms_thresh, score_thresh, detections_per_img,
         use_ocm_masks, fourth_band, replace, nir_init, use_wandb):

    os.makedirs(outputdir, exist_ok=True)

    channel_kinds = resolve_channels(fourth_band, replace)
    n_channels = len(channel_kinds)

    # Per-channel normalization. NDVI is fed raw and normalized with stats
    # derived from the training split; the other kinds use fixed stats.
    ndvi_mean = ndvi_std = None
    if 'ndvi' in channel_kinds:
        mask_files = sorted(glob.glob(os.path.join(imagedir, "*.mask.png")))
        train_img_files = [
            mf.replace('.mask.png', '.tif') for mf in mask_files
        ]
        ndvi_mean, ndvi_std = compute_ndvi_stats(
            train_img_files, 'left', size,
        )
        print(
            f"Derived NDVI normalization from training split: "
            f"mean={ndvi_mean:.4f} std={ndvi_std:.4f}"
        )
    image_mean = []
    image_std = []
    for kind in channel_kinds:
        if kind == 'ndvi':
            image_mean.append(ndvi_mean)
            image_std.append(ndvi_std)
        else:
            m, s = _KIND_NORM[kind]
            image_mean.append(m)
            image_std.append(s)

    params = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'size': size,
        'min_instance_size': min_instance_size,
        'nms_thresh': nms_thresh,
        'score_thresh': score_thresh,
        'detections_per_img': detections_per_img,
        'use_ocm_masks': use_ocm_masks,
        'fourth_band': fourth_band,
        'replace': replace,
        'channel_kinds': channel_kinds,
        'n_channels': n_channels,
        'image_mean': image_mean,
        'image_std': image_std,
        'ndvi_mean': ndvi_mean,
        'ndvi_std': ndvi_std,
        'nir_init': nir_init,
    }

    run = None
    if use_wandb:
        run = wandb.init(
            entity='tree-flower', project='planet-maskrcnn',
            config={
                'imagedir': imagedir,
                'outputdir': outputdir,
                **params,
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PlanetMaskRCNNDataset(
        imagedir, split='left', size=size, color_jitter=True,
        min_instance_size=min_instance_size,
        use_ocm_masks=use_ocm_masks, channel_kinds=channel_kinds,
    )
    test_dataset = PlanetMaskRCNNDataset(
        imagedir, split='right', size=size, color_jitter=False,
        min_instance_size=min_instance_size,
        use_ocm_masks=use_ocm_masks, channel_kinds=channel_kinds,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
    )

    model = build_model(
        num_classes=2, nms_thresh=nms_thresh, score_thresh=score_thresh,
        detections_per_img=detections_per_img,
        use_ocm_masks=use_ocm_masks, image_mean=image_mean,
        image_std=image_std, nir_init=nir_init,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    iou_metric = torchmetrics.JaccardIndex(task="binary").to(device)
    map_metric = MeanAveragePrecision(iou_type="segm").to(device)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_totals = {}
        n_batches = 0
        for imgs, targets in tqdm(train_loader, leave=False):
            imgs = [img.to(device) for img in imgs]
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                # OCM filtering can leave a fully-cloudy sample with no
                # surviving anchors/proposals, giving NaN losses; drop the
                # batch rather than corrupt the optimizer state.
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in loss_dict.items():
                loss_totals[k] = loss_totals.get(k, 0.0) + v.item()
            loss_totals['loss_total'] = (
                loss_totals.get('loss_total', 0.0) + loss.item()
            )
            n_batches += 1

        avg_losses = {k: v / max(n_batches, 1) for k, v in loss_totals.items()}

        test_iou, test_map, test_event = evaluate(
            model, test_loader, device, iou_metric, map_metric,
            use_ocm_masks=use_ocm_masks,
        )

        log = {'epoch': epoch, 'test_iou': test_iou}
        log.update({f'train/{k}': v for k, v in avg_losses.items()})
        for k, v in test_map.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                log[f'test_map/{k}'] = v.item()
        log.update({f'test_event/{k}': v for k, v in test_event.items()})
        print(
            f"Epoch {epoch+1}/{num_epochs} - loss: {avg_losses['loss_total']:.4f}"
            f" - test_iou: {test_iou:.4f}"
            f" - test_map50: {log.get('test_map/map_50', float('nan')):.4f}"
            f" - P/R/A: {test_event['precision']:.3f}"
            f"/{test_event['recall']:.3f}/{test_event['accuracy']:.3f}"
        )
        if run is not None:
            run.log(log)

        outputfile = os.path.join(outputdir, f'epoch_{epoch+1:03d}.pth')
        torch.save({'model': model, 'params': params}, outputfile)

    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
