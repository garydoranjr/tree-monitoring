#!/usr/bin/env python
"""Carry the last dated phantom crown polygons forward onto every mavic date.

The dated crown map (crowns_20240803/BCI_50ha_crownmap_timeseries.shp) ends
at 2024-03-18, the last phantom flight. extract_crown_stats_from_masks.py
needs a crown map with a `date` (YYYY_MM_DD) row for every mosaic it scores,
so this writes the --source-date polygons once per mavic classification
date. The output feeds crown_classifications_mavic.nc4.

Typical usage:
    python scripts/make_mavic_crown_dates.py \
        /Volumes/Earth03/flower/results/crowns_mavic_dates.gpkg
"""
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

FLOWER = Path('/Volumes/Earth03/flower')


@click.command()
@click.argument('output', type=click.Path(dir_okay=False, path_type=Path))
@click.option('--crownmap', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=FLOWER / 'crowns_20240803/BCI_50ha_crownmap_timeseries.shp',
              show_default=True)
@click.option('--source-date', default='2024_03_18', show_default=True,
              help='Dated-map date whose polygons are carried forward.')
@click.option('--classifications', type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=FLOWER / 'results/classifications', show_default=True,
              help='Directory of *_M3M_aligned_global_RGB_classifications.tif.')
def main(output, crownmap, source_date, classifications):
    crowns = gpd.read_file(crownmap)
    src = crowns[crowns['date'] == source_date]
    if src.empty:
        raise click.BadParameter(f'no crowns dated {source_date} in {crownmap}')
    dates = sorted(p.name.split('BCI_50ha_')[1][:10] for p in
                   classifications.glob('*_M3M_aligned_global_RGB_classifications.tif'))
    out = gpd.GeoDataFrame(pd.concat([src.assign(date=d) for d in dates], ignore_index=True),
                           crs=crowns.crs)
    out.to_file(output, driver='GPKG')
    click.echo(f'Wrote {output}: {len(src):,} crowns from {source_date} x {len(dates)} '
               f'mavic dates ({dates[0]} .. {dates[-1]})')


if __name__ == '__main__':
    main()
