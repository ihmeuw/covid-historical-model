from pathlib import Path
from collections import namedtuple
from loguru import logger

import pandas as pd

from covid_historical_model.rates import idr

RESULTS = namedtuple('Results', 'seroprevalence model_data mr_model_dict pred_location_map pred pred_fe')


def runner(model_inputs_root: Path, testing_root: Path,
           seroprevalence: pd.DataFrame,
           ifr_data: pd.Series,
           pred_start_date: str = '2020-01-01',
           pred_end_date: str = '2021-12-31',
           verbose: bool = True,) -> namedtuple:
    pred_start_date = pd.Timestamp(pred_start_date)
    pred_end_date = pd.Timestamp(pred_end_date)

    input_data = idr.data.load_input_data(model_inputs_root, testing_root,
                                          seroprevalence, verbose=verbose)
    model_data = idr.data.create_model_data(ifr_data=ifr_data, verbose=verbose, **input_data)
    pred_data = idr.data.create_pred_data(
        pred_start_date=pred_start_date, pred_end_date=pred_end_date,
        **input_data
    )
    
    # check what NAs in pred data might be about, get rid of them in safer way
    mr_model_dict, prior_dicts, pred, pred_fe, pred_location_map = idr.model.run_model(
        model_data=model_data.copy(),
        pred_data=pred_data.copy(),
        verbose=verbose,
        **input_data
    )
    
    rmse_data, floor_data = idr.flooring.find_idr_floor(
        pred=pred.copy(),
        daily_cases=input_data['daily_cases'].copy(),
        serosurveys=(seroprevalence
                     .set_index(['location_id', 'date'])
                     .sort_index()
                     .loc[:, 'seroprevalence']),
        population=input_data['population'].copy(),
        hierarchy=input_data['hierarchy'].copy(),
        test_range=[0.1] + list(range(1, 11)),
        verbose=verbose,
    )
    
    pred = (pred.reset_index().set_index('location_id').join(floor_data, how='left'))
    pred = (pred
            .reset_index()
            .groupby('location_id')
            .apply(lambda x: idr.flooring.rescale_idr(x.set_index('date').loc[:, 'pred_idr'],
                                                      x['idr_floor'].unique().item(),
                                                      ceiling=1.,))
            .rename('pred_idr'))

    dates_data = idr.model.determine_mean_date_of_infection(
        location_dates=model_data[['location_id', 'date']].drop_duplicates().values.tolist(),
        daily_cases=input_data['daily_cases'].copy(),
        pred=pred.copy()
    )
    model_data = model_data.merge(dates_data, how='left')
    model_data['avg_date_of_infection'] = model_data['avg_date_of_infection'].fillna(model_data['date'])
    model_data = (model_data.loc[:, ['location_id', 'avg_date_of_infection', 'idr']].reset_index(drop=True))
    model_data = model_data.rename(columns={'avg_date_of_infection':'date'})

    results = RESULTS(
        seroprevalence=input_data['seroprevalence'],
        model_data=model_data,
        mr_model_dict=mr_model_dict,
        pred_location_map=pred_location_map,
        pred=pred,
        pred_fe=pred_fe,
    )

    return results
