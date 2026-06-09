#!/usr/bin/env python
import os
import glob
import wandb
import click
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy import ndimage
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.ops import masks_to_boxes
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
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


def get_split(img, mask, split, size):
    img = np.asarray(img)
    mask = np.asarray(mask)

    if img.ndim < 2 or mask.ndim < 2:
        raise ValueError("img and mask must have at least 2 dimensions")
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("img and mask must match in their first two dimensions")

    h, w = img.shape[:2]
    row_start, row_end, col_start, col_end = _split_window(h, w, split, size)

    img_crop = img[row_start:row_end, col_start:col_end, ...]
    mask_crop = mask[row_start:row_end, col_start:col_end, ...]
    return img_crop, mask_crop


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


class PlanetMaskRCNNDataset(Dataset):

    def __init__(self, img_dir, split, size=512, color_jitter=False,
                 min_instance_size=4):
        self.mask_files = sorted(glob.glob(os.path.join(img_dir, "*.mask.png")))
        self.img_files = [
            mf.replace('.mask.png', '.png') for mf in self.mask_files
        ]
        self.split = split
        self.size = size
        self.min_instance_size = min_instance_size
        self.jitter = (
            T.ColorJitter(brightness=0.2, contrast=0.2) if color_jitter else None
        )
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_files[idx]))[..., :3]
        mask = np.array(Image.open(self.mask_files[idx]))
        mask = (mask == 255).astype(np.uint8)

        img_crop, mask_crop = get_split(img, mask, self.split, self.size)

        inst_masks = binary_mask_to_instances(
            mask_crop, min_instance_size=self.min_instance_size,
        )

        img_pil = Image.fromarray(img_crop)
        if self.jitter is not None:
            img_pil = self.jitter(img_pil)
        img_tensor = self.to_tensor(img_pil)  # (3, H, W) float32 in [0,1]

        if inst_masks.shape[0] == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, self.size, self.size), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
            return img_tensor, target

        masks_t = torch.from_numpy(inst_masks)  # (N, H, W) uint8
        boxes = masks_to_boxes(masks_t)         # (N, 4) float, inclusive xmax/ymax
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        masks_t = masks_t[keep]
        boxes = boxes[keep]
        if masks_t.shape[0] == 0:
            return img_tensor, {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, self.size, self.size), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
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
        return img_tensor, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def build_model(num_classes=2, nms_thresh=0.3, score_thresh=0.05,
                detections_per_img=300):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

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
    return model


@torch.no_grad()
def evaluate(model, dataloader, device, iou_metric, map_metric,
             score_thresh=0.5, iou_thresh=0.5):
    """Compute binary-IoU (union of masks), mask mAP, and event-level
    precision/recall/accuracy on the dataloader."""
    model.eval()
    iou_metric.reset()
    map_metric.reset()

    tp = 0
    fp = 0
    fn = 0

    for imgs, targets in dataloader:
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)

        preds_for_map = []
        for out in outputs:
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

        for pred, tgt in zip(preds_for_map, gts_for_map):
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
@click.option('--wandb/--no-wandb', 'use_wandb', default=True)
def main(imagedir, outputdir, num_epochs, batch_size, lr, size,
         min_instance_size, nms_thresh, score_thresh, detections_per_img,
         use_wandb):

    os.makedirs(outputdir, exist_ok=True)

    params = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'size': size,
        'min_instance_size': min_instance_size,
        'nms_thresh': nms_thresh,
        'score_thresh': score_thresh,
        'detections_per_img': detections_per_img,
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
    )
    test_dataset = PlanetMaskRCNNDataset(
        imagedir, split='right', size=size, color_jitter=False,
        min_instance_size=min_instance_size,
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
