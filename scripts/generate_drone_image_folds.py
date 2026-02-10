#!/usr/bin/env python
import os
import json
import click
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import GroupShuffleSplit


from generate_drone_image_masks import update_labels


def make_group_splits(df, id_col="polygon_id", group_col="polyid", 
                      train_size=0.8, test_size=0.2,
                      random_state=42):
    """
    Assigns each unique_id in df to train/test ensuring groups stay together.
    Saves result to CSV with columns [unique_id, split].
    """
    assert abs(train_size + test_size - 1.0) < 1e-6, "Splits must sum to 1"

    groups = df[group_col].unique()

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df[id_col], groups=df[group_col]))

    df_out = df[[id_col]].copy()

    # Initialize new column
    df_out["split"] = None

    # Assign values based on index
    df_out.loc[train_idx, "split"] = "train"
    df_out.loc[test_idx, "split"] = "test"

    return df_out


@click.command()
@click.argument('labelfile')
@click.argument('outputfile')
def main(labelfile, outputfile):

    labels = gpd.read_file(labelfile, layer='flowering_dataset')
    labels = update_labels(labels)

    labels['polyid'] = labels['polygon_id'].apply(lambda i: i.split('_')[0])

    df_out = make_group_splits(labels)
    df_out.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
