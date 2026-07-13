# Vendored from setup-sw-db/core/parse.py @ de72933 on 2026-04-19 — DO NOT EDIT.
# Subset retained: _dt_hpo, HP30 spec, parse_hpo.
# Re-sync: see src/_vendor/README.md.
"""Parser for the GFZ Potsdam Hp30/ap30 nowcast text file."""
from datetime import datetime

import numpy as np
import pandas as pd


def _dt_hpo(year, month, day, hh_start):
    """Build datetime from HPo record fields.

    Args:
        year: Year value.
        month: Month value.
        day: Day value.
        hh_start: Starting hour as float (e.g. 0.0, 0.5, 1.0).

    Returns:
        datetime, or pd.NaT on failure.
    """
    try:
        hours = int(hh_start)
        minutes = int(round((hh_start - hours) * 60))
        return datetime(int(year), int(month), int(day), hours, minutes)
    except (ValueError, TypeError):
        return pd.NaT


HP30 = {
    'table': 'hpo_hp30',
    'raw_columns': [
        'Year', 'Month', 'Day', 'hh_start', 'hh_mid',
        'days_start', 'days_mid', 'Hp30', 'ap30', 'D',
    ],
    'keep_columns': ['Year', 'Month', 'Day', 'hh_start', 'Hp30', 'ap30'],
    'fill_values': {'Hp30': -1.000, 'ap30': -1},
    'hp_key': 'Hp30',
    'ap_key': 'ap30',
}


def parse_hpo(text: str, spec: dict) -> pd.DataFrame:
    """Parse HPo (Hp30/Hp60) blank-separated text data into a DataFrame.

    Args:
        text: Raw text data with # comment headers.
        spec: Dict with keys 'raw_columns', 'keep_columns', 'fill_values'.

    Returns:
        DataFrame with datetime column and selected data columns.
    """
    from io import StringIO

    df = pd.read_csv(
        StringIO(text),
        comment='#',
        sep=r'\s+',
        header=None,
        names=spec['raw_columns'],
    )

    df = df[spec['keep_columns']].copy()

    for col, fill_val in spec['fill_values'].items():
        df[col] = df[col].replace(fill_val, np.nan)

    df['datetime'] = df.apply(
        lambda row: _dt_hpo(row['Year'], row['Month'], row['Day'], row['hh_start']),
        axis=1,
    )

    cols = ['datetime'] + [c for c in df.columns if c != 'datetime']
    df = df[cols]
    return df
