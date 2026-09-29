"""Score two or more Mask R-CNN checkpoints on one common test set.

Motivation: comparing training runs by their wandb summary metrics is invalid
when the runs used different `imagedir`s, because the left/right split means a
larger dataset also enlarges the *test* split. A recall-driven metric then
drops when harder positives are added to the evaluation set, with no change in
model quality.

This script pins the evaluation to an explicit set of chips (``--chip-list``,
typically the intersection of the two datasets' chip filenames) and to one
chip directory, so every checkpoint sees byte-identical inputs and identical
ground truth. Test crops only ('right' split) -- the same region the training
runs held out -- so no model is scored on pixels it trained on.

Example:

    python scripts/compare_maskrcnn_runs.py \\
        --chip-dir /scratch/.../20260706_full_..._curated \\
        --checkpoint baseline=20260824_maskrcnn_out_min064/epoch_012.pth \\
        --checkpoint extended=202609_full_curated_maskrcnn_out_min064/epoch_003.pth \\
        --ocm-masks
"""

import json
import os
import sys

import click
import torch
import torchmetrics
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection import MeanAveragePrecision

import train_planet_image_maskrcnn as tp
from train_planet_image_maskrcnn import (
    OCMMaskRCNN,
    OCMRoIHeads,
    OCMRPN,
    PlanetMaskRCNNDataset,
    collate_fn,
    evaluate,
    select_device,
)

# The checkpoints pickle whole model objects, and were written while the
# training script was running as __main__ -- so the custom classes are recorded
# as `__main__.OCMMaskRCNN` etc. Re-export them into this module's __main__ so
# the unpickler can resolve them here.
for _cls in (OCMMaskRCNN, OCMRoIHeads, OCMRPN, PlanetMaskRCNNDataset):
    setattr(sys.modules['__main__'], _cls.__name__, _cls)

METRIC_KEYS = ('test_iou', 'test_map/map', 'test_map/map_50', 'test_map/map_75')


def load_checkpoint(path, device):
    """Load a checkpoint saved by train_planet_image_maskrcnn.py.

    Checkpoints pickle the whole model object (not a state_dict), so
    weights_only=False is required.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ckpt['model'].to(device)
    model.eval()
    return model, ckpt.get('params', {}), ckpt.get('epoch')


def build_subset(chip_dir, chip_names, size, min_instance_size,
                 use_ocm_masks, channel_kinds):
    """Test-split dataset over chip_dir restricted to `chip_names`.

    Restricting by basename (rather than by index) keeps the evaluation set
    stable even if the two chip dirs enumerate different numbers of files.
    """
    ds = PlanetMaskRCNNDataset(
        chip_dir, split='right', size=size, color_jitter=False,
        min_instance_size=min_instance_size, use_ocm_masks=use_ocm_masks,
        channel_kinds=channel_kinds,
    )
    if chip_names is None:
        return ds, [os.path.basename(f) for f in ds.img_files]

    available = {os.path.basename(f): i for i, f in enumerate(ds.img_files)}
    missing = sorted(set(chip_names) - set(available))
    if missing:
        raise click.ClickException(
            f'{len(missing)} chip(s) from --chip-list are not in {chip_dir}, '
            f'e.g. {missing[:3]}'
        )
    keep = sorted(available[n] for n in chip_names)
    return Subset(ds, keep), [os.path.basename(ds.img_files[i]) for i in keep]


@click.command()
@click.option('--chip-dir', required=True,
              type=click.Path(exists=True, file_okay=False),
              help='Chip directory to evaluate on. Must contain every chip in '
                   '--chip-list. Use the smaller/older dataset when comparing '
                   'a dataset against its own superset.')
@click.option('--checkpoint', 'checkpoints', multiple=True, required=True,
              metavar='LABEL=PATH',
              help='A checkpoint to score, as label=path. Repeatable.')
@click.option('--chip-list', type=click.Path(exists=True, dir_okay=False),
              help='File of chip basenames (one per line, e.g. '
                   'foo_4band.tif) restricting the evaluation set. Without '
                   'it, every chip in --chip-dir is used.')
@click.option('--size', default=512, type=int)
@click.option('--min-instance-size', default=64, type=int,
              help='Must match the value used at training time, since it '
                   'filters the ground-truth crowns being scored.')
@click.option('--ocm-masks/--no-ocm-masks', 'use_ocm_masks', default=False)
@click.option('--score-thresh', default=0.5, type=float,
              help="Mask binarization threshold inside evaluate() (its own "
                   "`score_thresh`), not the model's detection threshold.")
@click.option('--iou-thresh', default=0.5, type=float)
@click.option('--json-out', type=click.Path(dir_okay=False),
              help='Write the full results table here as JSON.')
def main(chip_dir, checkpoints, chip_list, size, min_instance_size,
         use_ocm_masks, score_thresh, iou_thresh, json_out):
    parsed = []
    for spec in checkpoints:
        if '=' not in spec:
            raise click.ClickException(
                f'--checkpoint must be label=path, got {spec!r}'
            )
        label, path = spec.split('=', 1)
        if not os.path.exists(path):
            raise click.ClickException(f'no such checkpoint: {path}')
        parsed.append((label, path))

    chip_names = None
    if chip_list:
        with open(chip_list) as fh:
            chip_names = [ln.strip() for ln in fh if ln.strip()]

    device = select_device()
    iou_metric = torchmetrics.JaccardIndex(task='binary').to(device)
    map_metric = MeanAveragePrecision(iou_type='segm').to(device)

    results = {}
    n_eval = None
    for label, path in parsed:
        model, params, epoch = load_checkpoint(path, device)
        # Channel layout is a property of the trained model, so read it from
        # the checkpoint rather than assuming RGB.
        channel_kinds = params.get('channel_kinds', ('red', 'green', 'blue'))
        dataset, used = build_subset(
            chip_dir, chip_names, size, min_instance_size, use_ocm_masks,
            channel_kinds,
        )
        if n_eval is None:
            n_eval = len(used)
            print(f'Evaluating on {n_eval} chip(s) from {chip_dir} '
                  f'(right/test crops)\n')
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
        )
        iou, test_map, event = evaluate(
            model, loader, device, iou_metric, map_metric,
            score_thresh=score_thresh, iou_thresh=iou_thresh,
            use_ocm_masks=use_ocm_masks,
        )
        row = {'checkpoint': path, 'epoch': epoch, 'n_chips': len(used),
               'test_iou': iou, 'channel_kinds': list(channel_kinds)}
        for k, v in test_map.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                row[f'test_map/{k}'] = v.item()
        row.update({f'test_event/{k}': v for k, v in event.items()})
        results[label] = row
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    width = max(len(l) for l in results)
    header = f'{"run":<{width}}  ' + '  '.join(f'{k:>16s}' for k in METRIC_KEYS)
    print(header)
    print('-' * len(header))
    for label, row in results.items():
        cells = '  '.join(f'{row.get(k, float("nan")):16.4f}'
                          for k in METRIC_KEYS)
        print(f'{label:<{width}}  {cells}')

    print()
    ev = ('test_event/precision', 'test_event/recall', 'test_event/tp',
          'test_event/fp', 'test_event/fn')
    header = f'{"run":<{width}}  ' + '  '.join(f'{k.split("/")[1]:>10s}'
                                               for k in ev)
    print(header)
    print('-' * len(header))
    for label, row in results.items():
        cells = '  '.join(f'{row.get(k, 0):10.4f}' if 'ision' in k or 'call' in k
                          else f'{row.get(k, 0):10.0f}' for k in ev)
        print(f'{label:<{width}}  {cells}')

    if json_out:
        with open(json_out, 'w') as fh:
            json.dump({'chip_dir': chip_dir, 'n_chips': n_eval,
                       'use_ocm_masks': use_ocm_masks,
                       'min_instance_size': min_instance_size,
                       'results': results}, fh, indent=2)
        print(f'\nWrote {json_out}')


if __name__ == '__main__':
    main()
