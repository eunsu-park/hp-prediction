"""DB-free 30-minute aggregation of 1-minute solar wind data.

Mirrors the resample + min/avg/max flattening logic of
`setup-sw-db/core/aggregate.py::aggregate_sw_30min`, but consumes a DataFrame
directly so the real-time pipeline can skip PostgreSQL.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


# Source columns produced by fetch.noaa_swpc.fetch_swpc, in canonical order.
_SOURCE_COLS = ["v", "np", "t", "bx", "by", "bz", "bt"]

# Pandas uses "mean"; training schema uses "avg". Training order is avg/min/max
# for each physical variable (see regression-sw configs/base.yaml input_variables).
_AGG_ORDER = [("mean", "avg"), ("min", "min"), ("max", "max")]


def aggregate_30min(df_1min: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Resample 1-min solar wind data to 30-min avg/min/max per variable.

    Produces the 21 flattened columns expected by the training schema, in the
    exact order used by `configs/profile/base.yaml::data.timeseries.input_variables`
    (v_avg, v_min, v_max, np_avg, ..., bt_max).

    Args:
        df_1min: DataFrame with columns [datetime, v, np, t, bx, by, bz, bt].
            Missing columns are filled with NaN.
        start: Inclusive start of the target grid (30-min aligned).
        end: Inclusive end of the target grid (30-min aligned).

    Returns:
        DataFrame indexed by datetime (30-min) with 21 flattened columns.
        Rows missing from the input are returned as all-NaN.
    """
    grid = pd.date_range(start=start, end=end, freq="30min")
    if len(grid) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime"))

    working = df_1min.copy()
    for col in _SOURCE_COLS:
        if col not in working.columns:
            working[col] = pd.NA

    if "datetime" not in working.columns:
        raise ValueError("Input DataFrame must include a 'datetime' column")

    working["datetime"] = pd.to_datetime(working["datetime"])
    working = working.set_index("datetime").sort_index()

    # Restrict to the window plus one extra 30-min step so the final bin is complete.
    upper = end + pd.Timedelta(minutes=30)
    working = working.loc[(working.index >= start) & (working.index < upper), _SOURCE_COLS]

    # Coerce to numeric (NOAA occasionally returns string sentinel values).
    for col in _SOURCE_COLS:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    agg_dict = {col: ["mean", "min", "max"] for col in _SOURCE_COLS}
    resampled = working.resample("30min").agg(agg_dict)

    # Flatten columns to match training schema order: per-variable avg/min/max.
    flat_columns = []
    flat_data = {}
    for var in _SOURCE_COLS:
        for pandas_name, schema_name in _AGG_ORDER:
            flat = f"{var}_{schema_name}"
            flat_columns.append(flat)
            flat_data[flat] = resampled[(var, pandas_name)]

    result = pd.DataFrame(flat_data, index=resampled.index)
    result = result.reindex(grid)
    result.index.name = "datetime"

    logger.debug("Aggregated %d rows to %d 30-min bins", len(working), len(result))
    return result
