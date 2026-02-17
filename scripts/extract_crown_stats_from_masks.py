#!/usr/bin/env python
import os
import re
import click
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from tqdm import tqdm
import geopandas as gpd
from rasterstats import zonal_stats
from collections import defaultdict


DATE_PATTERN = re.compile(r'_(\d{4}_\d{2}_\d{2})_')


def get_date(path):
    m = DATE_PATTERN.search(os.path.basename(path))
    return m.group(1)


def get_stats(maskfile, crowns):

    crowns = crowns.dropna(subset=["geometry"])

    mask = rioxarray.open_rasterio(maskfile)

    stats = {}
    for i in range(mask.sizes["band"]):
        name = mask.attrs['long_name'][i]
        band = mask.isel(band=i)
        bstats = zonal_stats(
            crowns,
            np.squeeze(band.values),
            affine=band.rio.transform(),
            stats=['mean'],
            nodata=np.nan,
        )
        stats[name] = [ s['mean'] for s in bstats ]

    tags = crowns['tag']

    stat_dict = defaultdict(lambda: defaultdict(lambda: np.nan))

    for sname, slist in stats.items():
        for tag, s in zip(tags, slist):
            stat_dict[tag][sname] = s

    return set(stats.keys()), stat_dict


@click.command()
@click.argument('crownfile')
@click.argument('maskfiles', nargs=-1)
@click.argument('outputfile')
def main(crownfile, maskfiles, outputfile):

    crowns = gpd.read_file(crownfile)
    all_tags = sorted(crowns['tag'].unique())
    tag_to_latin = (
        crowns[["tag", "latin"]]
        .drop_duplicates(subset="tag")
        .set_index("tag")["latin"]
        .to_dict()
    )
    species_by_tag = np.array([
            tag_to_latin[t]
            for t in all_tags
        ], dtype=object
    )

    all_stat_names = set([])
    all_stats = {}
    for maskfile in tqdm(maskfiles):
        date = get_date(maskfile)
        relevant_crowns = crowns.loc[crowns['date'] == date]
        stat_names, stats = get_stats(maskfile, relevant_crowns)
        all_stat_names |= stat_names
        all_stats[date] = stats

    all_stat_names = sorted(all_stat_names)
    all_dates = sorted(all_stats.keys())

    data = np.array(
        [
            [
                [all_stats[d][t][n] for d in all_dates]
                for t in all_tags
            ]
            for n in all_stat_names
        ]
    )

    all_tags = np.array(all_tags, dtype=int)
    all_dates = np.array(
        [
            d.replace('_', '-') for d in all_dates
        ], dtype='datetime64[ns]'
    )

    data_vars = {
        band_name: (('tag', 'date'), data[i])
        for i, band_name in enumerate(all_stat_names)
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'tag': all_tags,
            'date': all_dates,
        }
    )
    ds['species'] = ('tag', species_by_tag)

    print(ds)
    ds.to_netcdf(outputfile, format='NETCDF4')


if __name__ == '__main__':
    main()
