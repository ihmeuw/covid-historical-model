from typing import Tuple, List
from tqdm import tqdm
from loguru import logger

import pandas as pd
import numpy as np

from covid_historical_model.utils.math import scale_to_bounds


def manual_floor_setting(rmse: pd.DataFrame,
                         best_floor: pd.Series,
                         hierarchy: pd.DataFrame,
                         data_locations: List[int],
                         verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    if verbose:
        logger.warning('Manually setting IDR floor of 0.1% for SSA locations.')
    is_ssa_location = hierarchy['path_to_top_parent'].apply(lambda x: '166' in x.split(','))
    ssa_location_ids = hierarchy.loc[is_ssa_location, 'location_id'].to_list()
    ssa_location_ids = [l for l in ssa_location_ids if l not in data_locations]
    
    for ssa_location_id in ssa_location_ids:
        if best_floor[ssa_location_id] > 0.001:
            best_floor[ssa_location_id] = 0.001
            is_ssa_rmse = rmse['location_id'] == ssa_location_id
            rmse.loc[is_ssa_rmse, 'rmse'] = np.nan
            rmse.loc[is_ssa_rmse, 'floor'] = 0.001

    return rmse, best_floor
    

def find_idr_floor(pred: pd.Series,
                   daily_cases: pd.Series,
                   serosurveys: pd.DataFrame,
                   population: pd.Series,
                   hierarchy: pd.DataFrame,
                   test_range: List,
                   verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:    
    rmse_floor = []
    for floor in test_range:
        rmse = test_floor_value(pred=pred.copy(),
                                daily_cases=daily_cases.copy(),
                                serosurveys=serosurveys.copy(),
                                population=population.copy(),
                                hierarchy=hierarchy.copy(),
                                floor=floor/100,
                                verbose=verbose,)
        rmse_floor.append(rmse.reset_index())
    rmse = pd.concat(rmse_floor).reset_index(drop=True)
    
    best_floor = rmse.groupby('location_id').apply(lambda x: x.sort_values('rmse')['floor'].values[0]).rename('idr_floor')
    
    rmse, best_floor = manual_floor_setting(rmse, best_floor, hierarchy,
                                            serosurveys.reset_index()['location_id'].unique().tolist(),
                                            verbose)
    
    return rmse, best_floor



def test_floor_value(pred: pd.Series,
                     daily_cases: pd.Series,
                     serosurveys: pd.Series,
                     population: pd.Series,
                     hierarchy: pd.DataFrame,
                     floor: float,
                     min_children: int = 3,
                     verbose: bool = True,) -> pd.DataFrame:
    if verbose:
        logger.info(f'Testing IDR floor of {round(floor*100, 2)}%.')
    
    pred = (pred
            .groupby(level=0)
            .apply(lambda x: scale_to_bounds(x, floor, 1.))
            .rename('idr'))
    
    daily_infections = (daily_cases / pred).dropna()
    daily_infections = daily_infections.dropna()
    cumulative_infections = daily_infections.groupby('location_id').cumsum()
    seroprevalence = (cumulative_infections / population).rename('seroprevalence')
    
    residuals = (seroprevalence - serosurveys).dropna().rename('residuals')
    
    rmses = pd.Series([],
                      name='rmse',
                      dtype='float',
                      index=pd.Index([], name='location_id'))
    location_ids = hierarchy.sort_values(['level', 'sort_order'])['location_id']
    for location_id in tqdm(location_ids):
        in_path = hierarchy['path_to_top_parent'].apply(lambda x: str(location_id) in x.split(','))
        child_ids = hierarchy.loc[in_path, 'location_id'].to_list()
        if location_id not in [95, 4749, 434]:
            # exclude England and Scotland if level above UK (too much data in those places; swamps algorithm)
            child_ids = [c for c in child_ids if c not in [4749, 434]]
        is_location = hierarchy['location_id'] == location_id
        parent_id = hierarchy.loc[is_location, 'parent_id'].item()
        # check if location_id is present
        if location_id in serosurveys.reset_index()['location_id'].to_list():
            rmse = np.sqrt((residuals[location_id]**2).mean())
        # check if at least `min_children` children are present
        elif serosurveys.reset_index()['location_id'].drop_duplicates().isin(child_ids).sum() >= min_children:
            child_residuals = residuals.reset_index()
            child_residuals = (child_residuals
                               .loc[child_residuals['location_id'].isin(child_ids)]
                               .set_index('date')
                               .loc[:, 'residuals'])
            rmse = np.sqrt((child_residuals**2).mean())
        # check if parent is present
        elif parent_id in rmses.index:
            rmse = rmses[parent_id]
        # we have a problem
        else:
            raise ValueError(f'No source of seroprevalence RMSE for location_id {location_id}.')
        rmse = pd.Series(rmse,
                         name='rmse',
                         index=pd.Index([location_id], name='location_id'))
        rmses = pd.concat([rmses, rmse])
    rmses = rmses.to_frame()
    rmses['floor'] = floor
    
    return rmses
