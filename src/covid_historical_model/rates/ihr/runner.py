from pathlib import Path
from collections import namedtuple
from loguru import logger

import pandas as pd

from covid_historical_model.rates import ihr

RESULTS = namedtuple('Results', 'seroprevalence model_data mr_model_dict pred_location_map pred pred_fe')


def runner(model_inputs_root: Path, age_pattern_root: Path,
           seroprevalence: pd.DataFrame,
           day_0: str = '2020-03-15',
           pred_start_date: str = '2020-01-01',
           pred_end_date: str = '2021-12-31',
           verbose: bool = True,) -> namedtuple:
    day_0 = pd.Timestamp(day_0)
    pred_start_date = pd.Timestamp(pred_start_date)
    pred_end_date = pd.Timestamp(pred_end_date)

    input_data = ihr.data.load_input_data(model_inputs_root, age_pattern_root,
                                          seroprevalence, verbose=verbose)
    model_data = ihr.data.create_model_data(day_0=day_0, **input_data)
    pred_data = ihr.data.create_pred_data(
        pred_start_date=pred_start_date, pred_end_date=pred_end_date,
        day_0=day_0, **input_data
    )
    
    # check what NAs in data might be about, get rid of them in safer way
    mr_model_dict, prior_dicts, pred, pred_fe, pred_location_map = ihr.model.run_model(
        model_data=model_data.copy(),
        pred_data=pred_data.copy(),
        verbose=verbose,
        **input_data
    )
    
    results = RESULTS(
        seroprevalence=input_data['seroprevalence'],
        model_data=model_data,
        mr_model_dict=mr_model_dict,
        pred_location_map=pred_location_map,
        pred=pred,
        pred_fe=pred_fe,
    )

    return results
