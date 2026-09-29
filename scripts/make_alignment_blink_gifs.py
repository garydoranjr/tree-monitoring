#!/usr/bin/env python
"""Blink GIFs of the Planet chip against its shift-corrected drone sidecar.

Each training chip carries a `.drone.png` sidecar: the drone ortho after the
single AROSICS shift, on an exact 2x subdivision of the chip grid. Blinking it
against the Planet preview shows the misalignment that survives the shift, and
therefore what the crown labels (`.mask.png`, drawn from the drone) inherit.
The crown-label outlines are drawn fixed on both frames, so they sit on the
drone crowns in one frame and show any offset against the Planet crowns in the
other.

Examples come from `measure_chip_local_offsets.py`'s window CSV (the
highest-offset well-correlated window in each of the worst chips per set), or
from explicit --stem/--center choices.

For each example this writes, into OUTDIR:
  blink_<n>_<stem>_zoom.gif   window around the chosen location, drone at its
                              native 2x resolution
  blink_<n>_<stem>_full.gif   whole chip, with the zoom window boxed
  blink_<n>_<stem>_pair.png   static side-by-side of the zoom frames plus a
                              red (Planet) / cyan (drone) overlay, for PDFs

Typical usage:
    python scripts/make_alignment_blink_gifs.py \\
        /Volumes/Earth03/flower/figs/202609_updates/residual_offsets_windows.csv \\
        /Volumes/Earth03/flower/figs/202609_updates/residual_offsets_chips.csv \\
        /Volumes/Earth03/flower/figs/202609_updates/gifs
"""
import json
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, gaussian_filter

Image.MAX_IMAGE_PIXELS = None

CHIP_RES_M = 0.75
OUTLINE_RGB = (255, 230, 0)
DATE_RE = re.compile(r'(\d{4})_(\d{2})_(\d{2})')


def font(size):
    for name in ('Helvetica.ttc', 'Arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def drone_date(chipdir, stem):
    """Drone flight date for a chip, from its coreg log record."""
    log = json.loads((Path(chipdir) / 'coreg_log.json').read_text())
    for rec in log:
        if rec['scene'] == stem and rec['coreg_ok']:
            m = DATE_RE.search(Path(rec['label']).name)
            return '-'.join(m.groups()), rec
    return '?', None


def planet_date(stem):
    return f'{stem[0:4]}-{stem[4:6]}-{stem[6:8]}'


def outline(mask):
    """One-pixel boundary of a binary mask."""
    return mask & ~binary_erosion(mask)


def paint_outline(img, mask):
    arr = np.asarray(img.convert('RGB')).copy()
    arr[outline(mask)] = OUTLINE_RGB
    return Image.fromarray(arr)


def caption_frame(img, title, subtitle, scale_px, scale_m, box=None):
    """Add a caption bar above the image and a scale bar on it."""
    w, h = img.size
    bar = 54
    out = Image.new('RGB', (w, h + bar), (255, 255, 255))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    d.text((8, 4), title, fill=(0, 0, 0), font=font(22))
    d.text((8, 30), subtitle, fill=(80, 80, 80), font=font(16))
    # Scale bar, bottom-left, on a dark backing for contrast.
    x0, y0 = 12, bar + h - 22
    d.rectangle([x0 - 4, y0 - 20, x0 + scale_px + 4, y0 + 10], fill=(0, 0, 0))
    d.rectangle([x0, y0, x0 + scale_px, y0 + 5], fill=(255, 255, 255))
    d.text((x0, y0 - 19), f'{scale_m:g} m', fill=(255, 255, 255), font=font(15))
    if box is not None:
        bx0, by0, bx1, by1 = box
        d.rectangle([bx0, by0 + bar, bx1, by1 + bar], outline=(0, 255, 255),
                    width=3)
    return out


def save_gif(frames, path, ms):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms,
                   loop=0, optimize=True)


def load_layers(chipdir, stem):
    base = Path(chipdir) / stem
    planet = Image.open(f'{base}.png').convert('RGB')
    drone2x = Image.open(f'{base}.drone.png').convert('RGB')
    mask = np.asarray(Image.open(f'{base}.mask.png')) > 127
    return planet, drone2x, mask


def make_example(n, row, outdir, zoom_px, full_width, ms, setname):
    chipdir, stem = row['chipdir'], row['stem']
    planet, drone2x, mask = load_layers(chipdir, stem)
    w, h = planet.size
    ddate, rec = drone_date(chipdir, stem)
    pdate = planet_date(stem)
    shift = ''
    if rec is not None and rec.get('x_shift_m') is not None:
        shift = (f"global shift applied: ({rec['x_shift_m']:+.1f}, "
                 f"{rec['y_shift_m']:+.1f}) m")

    # Zoom window, clamped to the chip.
    cx, cy = int(row['col']), int(row['row'])
    half = zoom_px // 2
    x0 = int(np.clip(cx - half, 0, w - zoom_px))
    y0 = int(np.clip(cy - half, 0, h - zoom_px))
    box = (x0, y0, x0 + zoom_px, y0 + zoom_px)

    # Zoom frames at 2x chip resolution: the drone at its native sidecar
    # resolution, the Planet chip and mask upsampled with nearest neighbour.
    z = 2
    p_zoom = planet.crop(box).resize((zoom_px * z,) * 2, Image.NEAREST)
    d_zoom = drone2x.crop(tuple(v * z for v in box))
    m_zoom = np.kron(mask[y0:y0 + zoom_px, x0:x0 + zoom_px],
                     np.ones((z, z), bool))
    scale_m = 50
    scale_px = int(round(scale_m / CHIP_RES_M * z))
    off = (f"measured residual here: {row['offset_m']:.1f} m "
           f"(dx {row['dx_m']:+.1f}, dy {row['dy_m']:+.1f})")
    fz = [caption_frame(paint_outline(p_zoom, m_zoom),
                        f'Planet {pdate}', off, scale_px, scale_m),
          caption_frame(paint_outline(d_zoom, m_zoom),
                        f'Drone {ddate} (after global shift)', shift or off,
                        scale_px, scale_m)]
    tag = f"blink_{n}_{stem.replace('_4band', '')}"
    save_gif(fz, outdir / f'{tag}_zoom.gif', ms)

    # Full-chip frames, downscaled for slide use.
    s = full_width / w
    size = (full_width, int(round(h * s)))
    p_full = planet.resize(size, Image.LANCZOS)
    d_full = drone2x.resize(size, Image.LANCZOS)
    m_full = np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST))
    sbox = tuple(int(round(v * s)) for v in box)
    scale_m_full = 200
    scale_px_full = int(round(scale_m_full / CHIP_RES_M * s))
    ff = [caption_frame(paint_outline(p_full, m_full), f'Planet {pdate}',
                        f'{setname}; yellow = crown labels, box = zoom',
                        scale_px_full, scale_m_full, sbox),
          caption_frame(paint_outline(d_full, m_full),
                        f'Drone {ddate} (after global shift)', shift,
                        scale_px_full, scale_m_full, sbox)]
    save_gif(ff, outdir / f'{tag}_full.gif', ms)

    # Static fallback: Planet | drone | red-cyan overlay.
    pg = np.asarray(p_zoom.convert('L'), dtype=np.float32)
    # Blur the drone to roughly the Planet resolution so the overlay compares
    # like with like instead of drone leaf texture against Planet pixels.
    dg = gaussian_filter(np.asarray(d_zoom.convert('L'), dtype=np.float32),
                         1.5 / CHIP_RES_M * z)

    def stretch(a):
        lo, hi = np.percentile(a, (2, 98))
        return np.clip((a - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)

    rc = Image.fromarray(np.dstack([stretch(pg), stretch(dg), stretch(dg)]))
    fo = caption_frame(rc, 'Overlay', 'red = Planet, cyan = drone',
                       scale_px, scale_m)
    pair = Image.new('RGB', (fz[0].width * 3 + 20, fz[0].height), 'white')
    for i, im in enumerate((fz[0], fz[1], fo)):
        pair.paste(im, (i * (fz[0].width + 10), 0))
    pair.save(outdir / f'{tag}_pair.png')
    return tag, pdate, ddate


@click.command()
@click.argument('windows_csv', type=click.Path(exists=True, dir_okay=False))
@click.argument('chips_csv', type=click.Path(exists=True, dir_okay=False))
@click.argument('outdir', type=click.Path(file_okay=False, path_type=Path))
@click.option('--per-set', default=3, show_default=True,
              help='Worst chips per set to render (ignored with --stem).')
@click.option('--min-ncc', default=0.5, show_default=True,
              help='Only pick windows whose match is at least this good.')
@click.option('--stem', 'stems', multiple=True,
              help='Render this chip (at its highest-offset window) instead '
                   'of automatic picks. Repeatable.')
@click.option('--zoom-px', default=256, show_default=True,
              help='Zoom window size in chip pixels (0.75 m).')
@click.option('--full-width', default=1000, show_default=True)
@click.option('--frame-ms', default=700, show_default=True)
def main(windows_csv, chips_csv, outdir, per_set, min_ncc, stems, zoom_px,
         full_width, frame_ms):
    outdir.mkdir(parents=True, exist_ok=True)
    wdf = pd.read_csv(windows_csv)
    cdf = pd.read_csv(chips_csv)[['stem', 'chipdir', 'spread_m']]
    wdf = wdf[wdf['ncc'] >= min_ncc].merge(cdf, on='stem')

    # Best window per chip = largest offset among well-matched windows.
    best = wdf.sort_values('offset_m', ascending=False).drop_duplicates('stem')
    if stems:
        picks = best[best['stem'].isin(stems)]
        missing = set(stems) - set(picks['stem'])
        if missing:
            raise click.ClickException(f'no qualifying windows for {missing}')
    else:
        picks = best.groupby('set', sort=False).head(per_set)

    rows = []
    for n, (_, row) in enumerate(picks.iterrows(), 1):
        tag, pdate, ddate = make_example(n, row, outdir, zoom_px, full_width,
                                         frame_ms, row['set'])
        sizes = {k: (outdir / f'{tag}_{k}.gif').stat().st_size / 1e6
                 for k in ('zoom', 'full')}
        click.echo(f"{tag}: {row['set']}, planet {pdate}, drone {ddate}, "
                   f"offset {row['offset_m']:.1f} m, ncc {row['ncc']:.2f}, "
                   f"zoom {sizes['zoom']:.1f} MB, full {sizes['full']:.1f} MB")
        rows.append({'tag': tag, 'set': row['set'], 'stem': row['stem'],
                     'planet_date': pdate, 'drone_date': ddate,
                     'offset_m': row['offset_m'], 'dx_m': row['dx_m'],
                     'dy_m': row['dy_m'], 'ncc': row['ncc'],
                     'chip_spread_m': row['spread_m']})
    pd.DataFrame(rows).to_csv(outdir / 'blink_examples.csv', index=False)


if __name__ == '__main__':
    main()
