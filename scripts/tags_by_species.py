#!/usr/bin/env python
import click
import geopandas as gpd


@click.command()
@click.argument("crownfile")
@click.argument("species")
def main(crownfile, species):
    """
    Print all tag IDs associated with a given species name.
    """

    crowns = gpd.read_file(crownfile)

    if "latin" not in crowns.columns:
        raise ValueError("Expected a 'latin' column in the crown file")

    subset = crowns[crowns["latin"] == species]

    if subset.empty:
        print(f"No tags found for species: {species}")
        return

    tags = sorted(subset["tag"].unique())

    print(f"Species: {species}")
    print(f"Number of tags: {len(tags)}")
    print("Tags:")
    for t in tags:
        print(f"  {t} {species.replace(' ', '_')}-{t}.png")


if __name__ == "__main__":
    main()

