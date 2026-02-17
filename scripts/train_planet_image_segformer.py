#!/usr/bin/env python
import os
import glob
import wandb
import click
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchmetrics
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


def get_split(img, mask, split, size):
    # Convert to numpy arrays (in case something array-like is passed)
    img = np.asarray(img)
    mask = np.asarray(mask)

    # --- Validation ---
    if img.ndim < 2 or mask.ndim < 2:
        raise ValueError("img and mask must have at least 2 dimensions")

    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("img and mask must match in their first two dimensions")

    h, w = img.shape[:2]

    if h < size:
        raise ValueError(f"Height must be at least {size}, got {h}")

    if w < 2 * size:
        raise ValueError(f"Width must be at least {2*size}, got {w}")

    if split not in ("left", "right"):
        raise ValueError('split must be either "left" or "right"')

    # --- Compute vertical slice (centered) ---
    center_row = h // 2
    row_start = center_row - size // 2
    row_end = row_start + size

    # Ensure bounds safety (in case of odd size interactions)
    if row_start < 0:
        row_start = 0
        row_end = size
    if row_end > h:
        row_end = h
        row_start = h - size

    # --- Compute horizontal slice ---
    center_col = w // 2

    if split == "left":
        col_end = center_col
        col_start = col_end - size
    else:  # "right"
        col_start = center_col
        col_end = col_start + size

    # Final safety check (should not trigger if constraints above hold)
    if col_start < 0 or col_end > w:
        raise ValueError("Computed window exceeds image bounds")

    # --- Extract windows ---
    img_crop = img[row_start:row_end, col_start:col_end, ...]
    mask_crop = mask[row_start:row_end, col_start:col_end, ...]

    return img_crop, mask_crop


class PlanetSegmentationDataset(Dataset):


    def __init__(self, img_dir, processor, split, size=512):
        self.mask_files = sorted(glob.glob(os.path.join(img_dir, "*.mask.png")))
        self.img_files = [
            mf.replace('.mask.png', '.png')
            for mf in self.mask_files
        ]
        self.processor = processor
        self.split = split
        self.size = size


    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, idx):
        # Load image (ignore 4th band)
        img = Image.open(self.img_files[idx])
        img = np.array(img)[..., :3]  # take only RGB

        # Load mask (convert 255 -> 1)
        mask = Image.open(self.mask_files[idx])
        mask = np.array(mask)
        mask = (mask == 255).astype(np.uint8)

        img, mask = get_split(img, mask, self.split, self.size)
        img = Image.fromarray(img)

        # Use processor to apply resizing, normalization
        encoded_inputs = self.processor(
            images=img,
            segmentation_maps=mask,
            size=self.size,
            return_tensors="pt"
        )

        pixel_values = encoded_inputs["pixel_values"].squeeze(0)  # (3,H,W)
        labels = encoded_inputs["labels"].squeeze(0)              # (H,W)

        return pixel_values, labels


def evaluate_segmentation(model, dataloader, metric, device="gpu", threshold=0.5):
    """
    Evaluate binary segmentation model with a given metric object.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        dataloader (torch.utils.data.DataLoader): Test/validation dataloader.
        metric (torchmetrics.Metric): A torchmetrics metric object (e.g., Dice, JaccardIndex).
        device (str): "gpu" or "cuda".
        threshold (float): Probability threshold to binarize predictions.

    Returns:
        float: Metric value
    """
    model.eval()
    metric = metric.to(device)
    if hasattr(metric, "reset"):  # torchmetrics supports reset()
        metric.reset()

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            output = model(images)
            logits = F.interpolate(
                output.logits, size=masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )

            # Sigmoid -> probabilities -> threshold
            #preds = torch.sigmoid(logits)
            #preds = (preds > threshold).int()
            preds = torch.argmax(logits, dim=1)

            # Update metric
            metric.update(preds, masks.int())

    return metric.compute().item()


@click.command()
@click.argument('imagedir')
@click.argument('outputdir')
def main(imagedir, outputdir):

    run = wandb.init(entity='tree-flower-planet', project='planet-segmentation')

    lr = 5e-5
    size = 512
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    processor = SegformerImageProcessor(do_resize=True, size=size, do_normalize=True)
    dataset = PlanetSegmentationDataset(
        imagedir,
        processor,
        split='left',
        size=size,
    )
    mask = dataset[0][1]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = PlanetSegmentationDataset(
        imagedir,
        processor,
        split='right',
        size=size,
    )
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Model: 2 classes (background + foreground)
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True  # allow head replacement
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    iou_metric = torchmetrics.JaccardIndex(task="binary").to(device)

    # ---------------------------
    # 3. Training loop
    # ---------------------------
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        model.train()
        for pixel_values, labels in tqdm(dataloader):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        train_iou = evaluate_segmentation(model, dataloader, iou_metric, device=device)
        test_iou = evaluate_segmentation(model, testloader, iou_metric, device=device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train IoU: {train_iou:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Test IoU: {test_iou:.4f}")

        run.log({
            'epoch': epoch,
            'train_iou': train_iou,
            'test_iou': test_iou,
            'train_loss': avg_loss,
        })

        outputfile = os.path.join(outputdir, f'epoch_{epoch+1:03d}.pth')
        torch.save(model, outputfile)

    run.finish()



if __name__ == '__main__':
    main()
