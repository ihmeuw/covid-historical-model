from typing import List

import pandas as pd


def fill_dates(data: pd.DataFrame, interp_vars: List[str]) -> pd.DataFrame:
    data = data.set_index('date').sort_index()
    data = data.asfreq('D').reset_index()
    data[interp_vars] = data[interp_vars].interpolate(axis=0)
    data['location_id'] = (data['location_id']
                           .fillna(method='pad')
                           .astype(int))

    return data[['location_id', 'date'] + interp_vars]


def str_fmt(str_col: pd.Series) -> pd.Series:
    fmt_str_col = str_col.copy()
    fmt_str_col = fmt_str_col.str.lower()
    fmt_str_col = fmt_str_col.str.strip()
    
    return fmt_str_col
