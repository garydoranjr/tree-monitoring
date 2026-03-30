#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


@click.command()
@click.argument('inputfile')
@click.argument('outputfile')
def main(inputfile, outputfile):
    df = pd.read_csv(inputfile, parse_dates=['date'])
    df['month'] = df['date'].dt.month

    # Compute median and IQR per month (combined across years)
    grouped = df.groupby('month')['fraction']
    stats = grouped.agg(
        median='median',
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75),
    ).reindex(range(1, 13))

    months = stats.index.values
    median = stats['median'].values
    q25 = stats['q25'].values
    q75 = stats['q75'].values

    with PdfPages(outputfile) as pdf:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.fill_between(months, q25, q75, alpha=0.3, color='steelblue', label='IQR (25–75%)')
        ax.plot(months, median, color='steelblue', linewidth=2, marker='o', label='Median')

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES)
        ax.set_xlim(1, 12)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Month')
        ax.set_ylabel('Fraction (> 0.5 confidence)')
        ax.set_title('Monthly Distribution of Fraction (Combined Across Years)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved plot to {outputfile}")


if __name__ == '__main__':
    main()
