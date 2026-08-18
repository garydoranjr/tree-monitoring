#!/usr/bin/env python
"""Visualize the --copy-paste augmentation of train_planet_image_maskrcnn.py.

Builds the same donor instance bank the trainer builds, draws augmented
samples from the same dataset class, and writes two figures: a per-chip
before/after comparison with the ground-truth instances outlined (original
vs pasted), and a zoomed view of individual pasted crowns beside the donor
patch they came from, so the paste can be checked pixel for pixel.

The RGB panels get a display-only 2-98 percentile stretch. The model input
uses the 0-99.9 stretch from build_input_channels, which on hazy Planet
chips is too flat to inspect by eye; only the display copy is restretched.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import click
import numpy as np
import matplotlib.pyplot as plt

from train_planet_image_maskrcnn import (
    PlanetMaskRCNNDataset,
    build_instance_bank,
    resolve_channels,
)


def _display_rgb(img):
    """(C, H, W) model input -> (H, W, 3) display RGB with a 2-98 percentile
    stretch, purely so faint crowns are visible on screen. The percentiles are
    shared across the three bands: stretching each band independently
    decorrelates the channels and turns canopy texture into color speckle."""
    rgb = np.asarray(img)[:3].transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(rgb, (2, 98))
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def _fill(rgb, mask, color, alpha=0.45):
    """Alpha-blend `color` into `rgb` wherever `mask` is set."""
    m = mask > 0
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.asarray(color, np.float32)
    return rgb


def _box(ax, mask, color, margin=6):
    """Draw a locator box around a mask, so a 10-pixel crown in a 512-pixel
    chip can actually be found by eye."""
    ys, xs = np.where(mask > 0)
    ax.add_patch(plt.Rectangle(
        (xs.min() - margin, ys.min() - margin),
        xs.max() - xs.min() + 2 * margin, ys.max() - ys.min() + 2 * margin,
        fill=False, edgecolor=color, linewidth=1.0,
    ))


@click.command()
@click.argument('imagedir')
@click.argument('outputdir')
@click.option('--split', default='left',
              type=click.Choice(['left', 'right', 'whole']),
              help="Crop to visualize. 'left' (default) is what training "
                   "uses; donors are always taken from this same crop.")
@click.option('--size', default=512, type=int)
@click.option('--min-instance-size', default=4, type=int)
@click.option('--fourth-band', default='none',
              type=click.Choice(['none', 'ir', 'ndvi']))
@click.option('--replace', default='none',
              type=click.Choice(['none', 'r', 'g', 'b']))
@click.option('--ocm-masks/--no-ocm-masks', 'use_ocm_masks', default=False,
              help='Load OCM cloud-mask sidecars, so pastes onto cloudy '
                   'pixels are rejected as they are during training.')
@click.option('--copy-paste-count', default=3, type=int)
@click.option('--copy-paste-prob', default=1.0, type=float,
              help='Defaults to 1.0 (unlike training) so every panel shows '
                   'pasted crowns.')
@click.option('--num-chips', default=4, type=int,
              help='Number of chips to show in the before/after figure.')
@click.option('--num-crowns', default=6, type=int,
              help='Number of pasted crowns to show zoomed.')
@click.option('--prefix', default='copy_paste',
              help='Output filename prefix, so several configurations can be '
                   'written into the same directory.')
@click.option('--seed', default=0, type=int)
def main(imagedir, outputdir, split, size, min_instance_size, fourth_band,
         replace, use_ocm_masks, copy_paste_count, copy_paste_prob, num_chips,
         num_crowns, prefix, seed):

    os.makedirs(outputdir, exist_ok=True)
    channel_kinds = resolve_channels(fourth_band, replace)
    print(f"Channel kinds: {channel_kinds}")

    bank = build_instance_bank(
        [(imagedir, split)], size, channel_kinds, min_instance_size,
        use_ocm_masks=use_ocm_masks,
    )
    if not bank:
        raise click.ClickException("No ground-truth crowns found to paste.")
    areas = np.array([int(d['mask'].sum()) for d in bank])
    print(f"Donor crown areas: min {areas.min()} median "
          f"{int(np.median(areas))} max {areas.max()} px")

    common = dict(split=split, size=size, color_jitter=False,
                  min_instance_size=min_instance_size,
                  use_ocm_masks=use_ocm_masks, channel_kinds=channel_kinds)
    ds_plain = PlanetMaskRCNNDataset(imagedir, **common)
    ds_cp = PlanetMaskRCNNDataset(
        imagedir, instance_bank=bank, copy_paste_count=copy_paste_count,
        copy_paste_prob=copy_paste_prob, **common)
    ds_cp.rng = np.random.default_rng(seed)

    n_chips = min(num_chips, len(ds_plain))
    fig, axes = plt.subplots(n_chips, 3, figsize=(13, 4.4 * n_chips),
                             squeeze=False)
    zoom_jobs = []
    for row in range(n_chips):
        plain_img, plain_tgt = ds_plain[row]
        cp_img, cp_tgt = ds_cp[row]
        n_orig = plain_tgt['masks'].shape[0]
        masks = cp_tgt['masks'].numpy()
        n_new = masks.shape[0] - n_orig

        axes[row][0].imshow(_display_rgb(plain_img))
        axes[row][0].set_title(f"original: {n_orig} labelled crowns",
                               fontsize=9)

        axes[row][1].imshow(_display_rgb(cp_img))
        for m in masks[n_orig:]:
            _box(axes[row][1], m, 'red')
        axes[row][1].set_title(
            f"copy-paste: +{n_new} crowns (boxed)", fontsize=9)

        ov = _display_rgb(cp_img)
        legend = "green = original GT, red = pasted GT"
        if use_ocm_masks:
            # Pastes are rejected over cloud, so show where that applied.
            cloudy = cp_tgt['clear_mask'].numpy() == 0
            _fill(ov, cloudy, [1, 1, 0], alpha=0.25)
            legend += f", yellow = cloud ({100 * cloudy.mean():.0f}%)"
        if n_orig:
            _fill(ov, masks[:n_orig].sum(axis=0), [0, 1, 0])
        if n_new:
            _fill(ov, masks[n_orig:].sum(axis=0), [1, 0, 0])
        axes[row][2].imshow(ov)
        axes[row][2].set_title(legend, fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        for m in masks[n_orig:]:
            zoom_jobs.append((cp_img, m))

    fig.suptitle(
        f"Copy-paste augmentation, {os.path.basename(imagedir.rstrip('/'))} "
        f"({split} crop, up to {copy_paste_count} crowns/chip from a "
        f"{len(bank)}-crown bank)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out1 = os.path.join(outputdir, f'{prefix}_examples.png')
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"Wrote {out1}")

    if not zoom_jobs:
        print("No crowns were pasted, so no zoom figure "
              "(try raising --copy-paste-prob or --copy-paste-count).")
        return

    n_z = min(num_crowns, len(zoom_jobs))
    fig, axes = plt.subplots(2, n_z, figsize=(2.6 * n_z, 6), squeeze=False)
    for col in range(n_z):
        img, mask = zoom_jobs[col]
        ys, xs = np.where(mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        half = 18
        y0 = min(max(cy - half, 0), mask.shape[0] - 2 * half)
        x0 = min(max(cx - half, 0), mask.shape[1] - 2 * half)
        sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        crop = _display_rgb(img)[sl]
        axes[0][col].imshow(crop)
        axes[0][col].set_title(f"pasted crown, {int(mask.sum())} px",
                              fontsize=9)
        ov = _fill(crop.copy(), mask[sl], [1, 0, 0])
        axes[1][col].imshow(ov)
        axes[1][col].set_title("with its instance mask", fontsize=9)
        for ax in (axes[0][col], axes[1][col]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Pasted crowns, zoomed. The mask outline must hug the "
                 "pasted pixels exactly.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out2 = os.path.join(outputdir, f'{prefix}_zoom.png')
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"Wrote {out2}")


if __name__ == '__main__':
    main()
