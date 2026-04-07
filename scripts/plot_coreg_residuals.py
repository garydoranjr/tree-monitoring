#!/usr/bin/env python
import os
import json
import click
import numpy as np
import warnings
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from werkzeug.security import safe_join

from coreg_global import load_offset_matrix, iterate, filterkeys


def compute_residuals(xy, offset):
    """
    Compute residual errors after applying global offsets.

    Parameters
    ----------
    xy : ndarray, shape (n, n, 2)
        Pairwise measured offsets
    offset : ndarray, shape (n, 2)
        Global offset for each image

    Returns
    -------
    residual : ndarray, shape (n, n, 2)
        Residual offsets for each pair
    magnitude : ndarray, shape (n, n)
        Magnitude of residual vectors
    """
    n = len(offset)
    residual = np.full((n, n, 2), np.nan)

    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(xy[i, j, 0]):
                continue

            # Remaining offset after applying global shifts
            # xy[i,j] + offset[j] ≈ offset[i] in optimal solution
            # So remaining offset = xy[i,j] - (offset[i] - offset[j])
            residual[i, j] = xy[i, j] + (offset[j] - offset[i])

    # Compute magnitude
    magnitude = np.sqrt(np.sum(residual**2, axis=2))

    return residual, magnitude


def print_statistics(residual, magnitude, keys):
    """
    Print diagnostic statistics about residual errors.

    Parameters
    ----------
    residual : ndarray, shape (n, n, 2)
        Residual offsets for each pair
    magnitude : ndarray, shape (n, n)
        Magnitude of residual vectors
    keys : list
        Image keys/names
    """
    # Mask diagonal and get valid residuals
    magnitude_masked = magnitude.copy()
    np.fill_diagonal(magnitude_masked, np.nan)
    valid_residuals = magnitude_masked[~np.isnan(magnitude_masked)]

    print("\n" + "="*60)
    print("Residual Alignment Error Statistics")
    print("="*60)
    print(f"Total valid pairs: {len(valid_residuals)}")
    print(f"Mean residual: {np.mean(valid_residuals):.3f} pixels")
    print(f"Median residual: {np.median(valid_residuals):.3f} pixels")
    print(f"Std residual: {np.std(valid_residuals):.3f} pixels")
    print(f"95th percentile: {np.percentile(valid_residuals, 95):.3f} pixels")
    print(f"99th percentile: {np.percentile(valid_residuals, 99):.3f} pixels")
    print(f"Max residual: {np.max(valid_residuals):.3f} pixels")

    # Identify worst pair
    worst_idx = np.unravel_index(np.nanargmax(magnitude_masked), magnitude_masked.shape)
    print(f"\nWorst pair:")
    print(f"  Images: {keys[worst_idx[0]]} -> {keys[worst_idx[1]]}")
    print(f"  Residual magnitude: {magnitude_masked[worst_idx]:.3f} pixels")
    print(f"  X residual: {residual[worst_idx][0]:.3f} pixels")
    print(f"  Y residual: {residual[worst_idx][1]:.3f} pixels")
    print("="*60 + "\n")


def plot_residual_heatmap(magnitude, keys, outputfile, figsize=12, dpi=150, vmax=None):
    """
    Create residual magnitude heatmap.

    Parameters
    ----------
    magnitude : ndarray, shape (n, n)
        Magnitude of residual vectors
    keys : list
        Image keys/names
    outputfile : str
        Output image path
    figsize : float
        Figure size in inches
    dpi : int
        Figure DPI
    vmax : float or None
        Max colorbar value (auto 99th percentile if None)
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # Mask diagonal
    magnitude_masked = magnitude.copy()
    np.fill_diagonal(magnitude_masked, np.nan)

    # Auto-scale if vmax not provided
    if vmax is None:
        vmax = np.nanpercentile(magnitude_masked, 99)

    # Create heatmap
    im = ax.imshow(
        magnitude_masked,
        cmap='plasma',
        interpolation='nearest',
        origin='upper',
        vmin=0,
        vmax=vmax
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Residual Magnitude (pixels)', fontsize=14)

    # Labels and title
    ax.set_xlabel('Target Image Index', fontsize=12)
    ax.set_ylabel('Source Image Index', fontsize=12)
    ax.set_title('Pairwise Residual Alignment Errors', fontsize=14, pad=20)

    # Conditional axis labels (only if not too many images)
    if len(keys) <= 50:
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=90, fontsize=6)
        ax.set_yticklabels(keys, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(outputfile, dpi=dpi)
    plt.close()


@click.command()
@click.argument('coregdir')
@click.argument('outputfile')
@click.option('-c', '--cloudfile', default=None, help='CSV file with cloud coverage data')
@click.option('-l', '--minclear', type=float, default=0.9, help='Minimum clear percentage')
@click.option('-m', '--maxiter', type=int, default=100, help='Maximum iterations')
@click.option('-t', '--tolerance', type=float, default=1e-6, help='Convergence tolerance')
@click.option('--figsize', type=float, default=12, help='Figure size in inches')
@click.option('--dpi', type=int, default=150, help='Figure DPI')
@click.option('--vmax', type=float, default=None, help='Max colorbar value in pixels')
def main(coregdir, outputfile, cloudfile, minclear, maxiter, tolerance, figsize, dpi, vmax):
    """
    Visualize residual alignment errors after global coregistration.

    Loads pairwise coregistration results, runs iterative global alignment,
    and creates a heatmap showing residual errors for each image pair.
    """
    # Load pairwise coregistration files
    files = sorted(glob(safe_join(coregdir, '*.json')))

    if len(files) == 0:
        raise ValueError(f"No JSON files found in {coregdir}")

    # Filter by cloud coverage if provided
    filtered = filterkeys(cloudfile, minclear)

    # Load offset matrix
    print(f"Loading {len(files)} coregistration files...")
    keys, xy = load_offset_matrix(files, keys=filtered)

    # Check for sufficient data
    valid_count = np.sum(~np.isnan(xy[:, :, 0]))
    if valid_count < 10:
        raise ValueError(f"Insufficient valid pairwise offsets: {valid_count}")

    if len(keys) > 1000:
        warnings.warn(f"Large matrix ({len(keys)}x{len(keys)}). Consider filtering with --minclear")

    print(f"Loaded {len(keys)} images with {valid_count} valid pairwise offsets")

    # Run iterative optimization
    offset = np.zeros((len(xy), 2))

    it = tqdm(list(range(maxiter)), 'Optimizing')
    for i in it:
        prev = np.array(offset)
        offset = iterate(offset, xy)
        delta = np.sqrt(np.nanmean(np.square(prev - offset)))
        it.set_postfix({'delta': f'{delta:.2e}'})
        if delta < tolerance:
            it.close()
            print(f"Converged after {i+1} iterations (delta={delta:.2e})")
            break
    else:
        warnings.warn(f"Did not converge after {maxiter} iterations (delta={delta:.2e})")

    # Compute residuals
    print("Computing residuals...")
    residual, magnitude = compute_residuals(xy, offset)

    # Check for all NaN residuals
    if np.all(np.isnan(magnitude)):
        raise ValueError("All residuals are NaN - check input data")

    # Print statistics
    print_statistics(residual, magnitude, keys)

    # Create heatmap
    print(f"Creating heatmap: {outputfile}")
    plot_residual_heatmap(magnitude, keys, outputfile, figsize, dpi, vmax)

    print(f"Done! Heatmap saved to {outputfile}")


if __name__ == '__main__':
    main()
