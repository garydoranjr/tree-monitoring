#!/usr/bin/env python
import yaml
import click
import asyncio
from pathlib import Path
from planet import Auth, Session

async def fetch(auth, orderid, outputdir):

    async with Session(auth=auth) as sess:
        client = sess.client('orders')

        # Download order
        await client.download_order(
            orderid, directory=outputdir, progress_bar=True
        )


@click.command()
@click.argument('orderid')
@click.argument('outputdir', type=click.Path(
    path_type=Path, exists=True
))
def main(orderid, outputdir):

    auth = Auth.from_env()
    asyncio.run(fetch(auth, orderid, outputdir))


if __name__ == '__main__':
    main()
