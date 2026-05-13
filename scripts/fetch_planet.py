#!/usr/bin/env python
import csv
import yaml
import click
import asyncio
from pathlib import Path
from datetime import datetime
from planet import (
    Auth, Session, data_filter, order_request,
    reporting
)


PLANET_ORDER_LIMIT = 100


async def search(auth, config, year, month):

    item_types = [config['item_type']]

    # Construct search filters
    conditions = [
        data_filter.permission_filter(),
        data_filter.date_range_filter(
            'acquired',
            gte=datetime(year, month, 1),
            lt=datetime(
                year if month < 12 else year + 1,
                month + 1 if month < 12 else 1,
                1
            ),
        )
    ]

    asset_types = config.get('asset_types', None)
    if asset_types is not None:
        conditions.append( data_filter.asset_filter(asset_types) )

    string_filters = config.get('string_filters', [])
    conditions += [
        data_filter.string_in_filter(**sf)
        for sf in string_filters
    ]

    geom = config.get('geometry', None)
    if geom is not None:
        conditions.append( data_filter.geometry_filter(geom) )

    sfilter = data_filter.and_filter(conditions)

    async with Session(auth=auth) as sess:
        client = sess.client('data')
        return [i async for i in client.search(item_types, sfilter)]


def read_missing_ids(csv_path):
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return [row[0] for row in reader if row and row[0]]


async def submit_order(auth, config, items, order_name, outputdir):

    item_type = config['item_type']
    product_bundle = config['product_bundle']

    geom = config.get('geometry', None)
    tools = [] if geom is None else [order_request.clip_tool(aoi=geom)]

    products = [order_request.product(
        item_ids=items,
        product_bundle=product_bundle,
        item_type=item_type
    )]

    request = order_request.build_request(
        name=order_name, products=products, tools=tools
    )

    async with Session(auth=auth) as sess:
        client = sess.client('orders')

        with reporting.StateBar(state='creating') as bar:

            # Create order
            order = await client.create_order(request)
            bar.update(state='created', order_id=order['id'])

            # Wait for order
            await client.wait(
                order['id'], max_attempts=0, callback=bar.update_state
            )

        # Download order
        await client.download_order(
            order['id'], directory=outputdir, progress_bar=True
        )


async def fetch(auth, config, year, month, outputdir):

    results = await search(auth, config, year, month)

    items = [r['id'] for r in results]

    if len(items) > PLANET_ORDER_LIMIT:
        raise ValueError(f'Too many items ({len(items)}) to fetch!')

    if len(items) == 0:
        raise ValueError('No items to fetch!')

    order_name = config['order_name_format'].format(year=year, month=month)
    await submit_order(auth, config, items, order_name, outputdir)


async def fetch_missing(auth, config, ids, order_name_prefix, outputdir):

    if len(ids) == 0:
        raise ValueError('No items to fetch!')

    chunks = [
        ids[i:i + PLANET_ORDER_LIMIT]
        for i in range(0, len(ids), PLANET_ORDER_LIMIT)
    ]

    for i, chunk in enumerate(chunks):
        if len(chunks) == 1:
            order_name = order_name_prefix
        else:
            order_name = f'{order_name_prefix}_{i + 1:02d}'
        click.echo(
            f'Submitting order {order_name} '
            f'({len(chunk)} items, {i + 1}/{len(chunks)})'
        )
        await submit_order(auth, config, chunk, order_name, outputdir)


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path, exists=True
))
@click.option('-y', '--year', type=int, default=None)
@click.option('-m', '--month', type=int, default=None)
@click.option('--missing-csv', type=click.Path(
    path_type=Path, exists=True
), default=None, help='CSV of scene IDs (first column) to fetch.')
@click.option('--order-name', type=str, default=None,
              help='Order name (or prefix when batched). '
                   'Required with --missing-csv.')
def main(configfile, outputdir, year, month, missing_csv, order_name):

    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

    auth = Auth.from_env()

    if missing_csv is not None:
        if order_name is None:
            raise click.UsageError(
                '--order-name is required when --missing-csv is given.'
            )
        ids = read_missing_ids(missing_csv)
        asyncio.run(fetch_missing(auth, config, ids, order_name, outputdir))
    else:
        if year is None or month is None:
            raise click.UsageError(
                '--year and --month are required '
                '(or use --missing-csv with --order-name).'
            )
        asyncio.run(fetch(auth, config, year, month, outputdir))


if __name__ == '__main__':
    main()
