#!/usr/bin/env python
import click
import numpy as np
import geopandas as gpd


@click.command()
@click.argument('labelfile')
@click.argument('outputfile')
def main(labelfile, outputfile):

    labels = gpd.read_file(labelfile, layer='flowering_dataset')
    labels['polyid'] = labels['polygon_id'].apply(lambda i: i.split('_')[0])
    labels['easting'] = labels['geometry'].apply(lambda g: g.centroid.x)

    threshold = np.quantile(labels['easting'].values, 0.8)

    labels['split'] = labels['easting'].apply(
        lambda e: 'train' if e < threshold else 'test'
    )

    df_out = labels[['polygon_id', 'split']]
    df_out.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
