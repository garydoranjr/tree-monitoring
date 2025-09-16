#!/usr/bin/env python
import os
import glob
import click
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class TifSegmentationDataset(Dataset):


    def __init__(self, img_dir, mask_dir, processor, size=512):
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        self.img_files = [
            os.path.join(img_dir, os.path.basename(mf))
            for mf in self.mask_files
        ]
        self.processor = processor
        self.size = size


    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, idx):
        # Load image (ignore 4th band)
        img = Image.open(self.img_files[idx])
        img = np.array(img)[..., :3]  # take only RGB
        img = Image.fromarray(img)

        # Load mask (convert 255 → 1)
        mask = Image.open(self.mask_files[idx])
        mask = np.array(mask)
        mask = (mask == 255).astype(np.uint8)

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


@click.command()
@click.argument('imagedir')
@click.argument('maskdir')
def main(imagedir, maskdir):

    lr = 5e-5
    size = 512
    num_epochs = 10
    device = 'cpu'
    processor = SegformerImageProcessor(do_resize=True, size=size, do_normalize=True)
    dataset = TifSegmentationDataset(imagedir, maskdir, processor, size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model: 2 classes (background + foreground)
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True  # allow head replacement
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Loss: CrossEntropy (model already outputs logits)
    criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # 3. Training loop
    # ---------------------------
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
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



if __name__ == '__main__':
    main()
