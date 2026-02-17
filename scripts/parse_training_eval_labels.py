#!/usr/bin/env python3

import json
import click
import pandas as pd


@click.command()
@click.argument("input_ndjson", type=click.Path(exists=True))
@click.argument("output_csv", type=click.Path())
def main(input_ndjson, output_csv):
    records = []

    with open(input_ndjson, "r") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            external_id = row.get("data_row", {}).get("external_id")
            quality_value = None

            projects = row.get("projects", {})
            for project_data in projects.values():
                for label in project_data.get("labels", []):
                    classifications = (
                        label.get("annotations", {})
                        .get("classifications", [])
                    )
                    for classification in classifications:
                        if classification.get("name") == "Quality":
                            radio = classification.get("radio_answer", {})
                            quality_value = radio.get("name")
                            break
                    if quality_value:
                        break
                if quality_value:
                    break

            records.append(
                {
                    "external_id": external_id,
                    "quality": quality_value,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    click.echo(f"Wrote {len(df)} rows to {output_csv}")


if __name__ == "__main__":
    main()

