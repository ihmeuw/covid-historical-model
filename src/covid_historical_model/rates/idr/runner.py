from typing import Dict
from pathlib import Path
from collections import namedtuple
from loguru import logger

import pandas as pd

from covid_historical_model.rates import idr
from covid_historical_model.rates import squeeze
from covid_historical_model.durations.durations import EXPOSURE_TO_CASE
from covid_historical_model.etl import model_inputs
from covid_historical_model.utils.math import scale_to_bounds

RESULTS = namedtuple('Results',
                     'seroprevalence testing_capacity model_data mr_model_dict pred_location_map level_lambdas ' \
                     'floor_data floor_rmse daily_numerator pred pred_fe')


def runner(input_data: Dict, shared: Dict,
           pred_ifr: pd.Series, daily_reinfection_inflation_factor: pd.Series,
           pred_start_date: str = '2019-11-01',
           pred_end_date: str = '2021-12-31',
           verbose: bool = True,) -> namedtuple:
    pred_start_date = pd.Timestamp(pred_start_date)
    pred_end_date = pd.Timestamp(pred_end_date)

    model_data = idr.data.create_model_data(pred_ifr=pred_ifr, verbose=verbose, **input_data)
    pred_data = idr.data.create_pred_data(
        pred_start_date=pred_start_date, pred_end_date=pred_end_date,
        **input_data
    )
    
    # check what NAs in pred data might be about, get rid of them in safer way
    mr_model_dict, prior_dicts, pred, pred_fe, pred_location_map, level_lambdas = idr.model.run_model(
        model_data=model_data.copy(),
        pred_data=pred_data.copy(),
        verbose=verbose,
        **input_data
    )
    
    rmse_data, floor_data = idr.flooring.find_idr_floor(
        pred=pred.copy(),
        daily_cases=input_data['daily_cases'].copy(),
        serosurveys=(input_data['seroprevalence']
                     .set_index(['location_id', 'date'])
                     .sort_index()
                     .loc[:, 'seroprevalence']).copy(),
        population=input_data['population'].copy(),
        hierarchy=input_data['adj_gbd_hierarchy'].copy(),
        test_range=[0.01, 0.1] + list(range(1, 11)),
        verbose=verbose,
    )
    
    pred = (pred.reset_index().set_index('location_id').join(floor_data, how='left'))
    pred = (pred
            .reset_index()
            .groupby('location_id')
            .apply(lambda x: scale_to_bounds(x.set_index('date').loc[:, 'pred_idr'],
                                             x['idr_floor'].unique().item(),
                                             ceiling=1.,))
            .rename('pred_idr'))
    
    pred = squeeze.squeeze(
        daily=input_data['daily_cases'].copy(),
        rate=pred.copy(),
        day_shift=EXPOSURE_TO_CASE,
        population=input_data['population'].copy(),
        daily_reinfection_inflation_factor=(daily_reinfection_inflation_factor
                                            .set_index(['location_id', 'date'])
                                            .loc[:, 'inflation_factor']
                                            .copy()),
        vaccine_coverage=input_data['vaccine_coverage'].copy(),
    )

    dates_data = idr.model.determine_mean_date_of_infection(
        location_dates=model_data[['location_id', 'date']].drop_duplicates().values.tolist(),
        daily_cases=input_data['daily_cases'].copy(),
        pred=pred.copy()
    )
    model_data = model_data.merge(dates_data, how='left')
    model_data['mean_infection_date'] = model_data['mean_infection_date'].fillna(model_data['date'])
    model_data = (model_data.loc[:, ['location_id', 'date', 'mean_infection_date', 'idr']].reset_index(drop=True))

    results = RESULTS(
        seroprevalence=input_data['seroprevalence'],
        testing_capacity=input_data['testing_capacity'],
        model_data=model_data,
        mr_model_dict=mr_model_dict,
        pred_location_map=pred_location_map,
        level_lambdas=level_lambdas,
        floor_data=floor_data,
        floor_rmse=rmse_data,
        daily_numerator=input_data['daily_cases'],
        pred=pred,
        pred_fe=pred_fe,
    )

    return results
