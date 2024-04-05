#!/usr/bin/env python
import yaml
import click
import asyncio
from pathlib import Path
from datetime import datetime
from planet import (
    Auth, Session, data_filter, order_request,
    reporting
)


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


async def fetch(auth, config, year, month, outputdir):

    results = await search(auth, config, year, month)

    items = [r['id'] for r in results]

    if len(items) > 100:
        raise ValueError('Too many items ({len(items)}) to fetch!')

    item_type = config['item_type']
    product_bundle = config['product_bundle']
    order_name = config['order_name_format'].format(year=year, month=month)

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


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path, exists=True
))
@click.option('-y', '--year', default=2021, type=int)
@click.option('-m', '--month', default=1, type=int)
def main(configfile, outputdir, year, month):

    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

    auth = Auth.from_env()
    asyncio.run(fetch(auth, config, year, month, outputdir))


if __name__ == '__main__':
    main()
