from pathlib import Path
from collections import namedtuple
from loguru import logger

import pandas as pd

from covid_historical_model.rates import ihr
from covid_historical_model.rates import post
from covid_historical_model.rates import squeeze
from covid_historical_model.durations.durations import EXPOSURE_TO_ADMISSION

RESULTS = namedtuple('Results',
                     'seroprevalence model_data mr_model_dict pred_location_map level_lambdas ' \
                     'daily_numerator pred pred_unadj pred_fe pred_lr pred_hr age_stand_scaling_factor')


def runner(model_inputs_root: Path, age_pattern_root: Path,
           seroprevalence: pd.DataFrame, vaccine_coverage: pd.DataFrame,
           escape_variant_prevalence: pd.Series,
           severity_variant_prevalence: pd.Series,
           reinfection_inflation_factor: pd.Series,
           day_0: str = '2020-03-15',
           pred_start_date: str = '2020-01-01',
           pred_end_date: str = '2021-12-31',
           verbose: bool = True,) -> namedtuple:
    day_0 = pd.Timestamp(day_0)
    pred_start_date = pd.Timestamp(pred_start_date)
    pred_end_date = pd.Timestamp(pred_end_date)

    input_data = ihr.data.load_input_data(model_inputs_root, age_pattern_root,
                                          seroprevalence, vaccine_coverage,
                                          escape_variant_prevalence,
                                          severity_variant_prevalence,
                                          verbose=verbose)
    model_data = ihr.data.create_model_data(day_0=day_0, **input_data)
    pred_data = ihr.data.create_pred_data(
        pred_start_date=pred_start_date, pred_end_date=pred_end_date,
        day_0=day_0, **input_data
    )
    
    # check what NAs in data might be about, get rid of them in safer way
    mr_model_dict, prior_dicts, pred, pred_fe, pred_location_map, age_stand_scaling_factor, level_lambdas = ihr.model.run_model(
        model_data=model_data.copy(),
        pred_data=pred_data.copy(),
        verbose=verbose,
        **input_data
    )
    pred_unadj = pred.copy()
    
    pred, pred_lr, pred_hr, *_ = post.variants_vaccines(
        rate_age_pattern=input_data['ihr_age_pattern'].copy(),
        denom_age_pattern=input_data['sero_age_pattern'].copy(),
        age_spec_population=input_data['age_spec_population'].copy(),
        rate=pred.copy(),
        day_shift=EXPOSURE_TO_ADMISSION,
        escape_variant_prevalence=input_data['escape_variant_prevalence'].copy(),
        severity_variant_prevalence=input_data['severity_variant_prevalence'].copy(),
        vaccine_coverage=input_data['vaccine_coverage'].copy(),
        population=input_data['population'].copy(),
    )
    
    lr_rr = pred_lr / pred
    hr_rr = pred_hr / pred
    pred = squeeze.squeeze(
        daily=input_data['daily_hospitalizations'].copy(),
        rate=pred.copy(),
        day_shift=EXPOSURE_TO_ADMISSION,
        population=input_data['population'].copy(),
        reinfection_inflation_factor=(reinfection_inflation_factor
                                      .set_index(['location_id', 'date'])
                                      .loc[:, 'inflation_factor']
                                      .copy()),
        vaccine_coverage=input_data['vaccine_coverage'].copy(),
    )
    pred_lr = lr_rr * pred
    pred_hr = hr_rr * pred

    
    results = RESULTS(
        seroprevalence=input_data['seroprevalence'],
        model_data=model_data,
        mr_model_dict=mr_model_dict,
        pred_location_map=pred_location_map,
        level_lambdas=level_lambdas,
        daily_numerator=input_data['daily_hospitalizations'].copy(),
        pred=pred,
        pred_unadj=pred_unadj,
        pred_fe=pred_fe,
        pred_lr=pred_lr,
        pred_hr=pred_hr,
        age_stand_scaling_factor=age_stand_scaling_factor,
    )

    return results
