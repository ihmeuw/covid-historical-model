import os
import sys
from contextlib import contextmanager

import pandas as pd


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
            
def parent_inheritance(data: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f'Needs to provide DataFrame, not {type(data)}.')
    if data.index.names[0] != 'location_id':
        raise ValueError('Index level 0 needs to be `location_id`.')
    location_ids = hierarchy['location_id'].to_list()
    path_to_top_parents = hierarchy['path_to_top_parent'].to_list()
    path_to_top_parents = [list(reversed(p.split(',')[:-1])) for p in path_to_top_parents]
    for location_id, path_to_top_parent in zip(location_ids, path_to_top_parents):
        if location_id not in data.reset_index()['location_id'].to_list():
            for parent_id in path_to_top_parent:
                try:
                    data = data.append(data.loc[int(parent_id)].rename(location_id))
                    break
                except KeyError:
                    pass
    
    return data
