#!/usr/bin/env python3

import shutil
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.argument("input_csv", type=click.Path(exists=True))
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
def main(input_csv, input_dir, output_dir):
    """
    Copy all files with quality == "Good" from input_dir to output_dir.
    Also copies corresponding .mask.png files if they exist.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if "external_id" not in df.columns or "quality" not in df.columns:
        raise click.ClickException(
            "CSV must contain 'external_id' and 'quality' columns."
        )

    good_rows = df[df["quality"] == "Good"]

    copied_count = 0

    for filename in good_rows["external_id"]:
        if pd.isna(filename):
            continue

        src_file = input_dir / filename
        dst_file = output_dir / filename

        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        else:
            click.echo(f"Warning: {src_file} not found", err=True)

        # Copy corresponding .mask.png file
        if filename.endswith(".png"):
            mask_name = filename[:-4] + ".mask.png"
            src_mask = input_dir / mask_name
            dst_mask = output_dir / mask_name

            if src_mask.exists():
                shutil.copy2(src_mask, dst_mask)
            else:
                # Not an error — masks may not exist
                pass

    click.echo(f"Copied {copied_count} Good files (plus any masks found).")


if __name__ == "__main__":
    main()

