import sys
import os
from pathlib import Path
from typing import Dict
import itertools
from collections import namedtuple
from loguru import logger
import dill as pickle

import pandas as pd

from covid_shared.cli_tools.logging import configure_logging_to_terminal

from covid_historical_model.rates import ifr
from covid_historical_model.rates import ihr
from covid_historical_model.rates import reinfection
from covid_historical_model.rates import serology
from covid_historical_model.rates import age_standardization
from covid_historical_model.rates import post
from covid_historical_model.rates import squeeze
from covid_historical_model.durations.durations import EXPOSURE_TO_DEATH

RESULTS = namedtuple('Results',
                     'seroprevalence model_data mr_model_dict pred_location_map daily_numerator level_lambdas ' \
                     'pred pred_unadj pred_fe pred_lr pred_hr pct_inf_lr pct_inf_hr age_stand_scaling_factor')


def runner(input_data: Dict,
           day_inflection: str,
           day_0: str = '2020-03-15',
           pred_start_date: str = '2019-11-01',
           pred_end_date: str = '2021-12-31',
           verbose: bool = True,) -> Dict:
    ## SET UP
    day_inflection = pd.Timestamp(day_inflection)
    day_0 = pd.Timestamp(day_0)
    pred_start_date = pd.Timestamp(pred_start_date)
    pred_end_date = pd.Timestamp(pred_end_date)

    model_data = ifr.data.create_model_data(day_0=day_0, **input_data)
    pred_data = ifr.data.create_pred_data(
        pred_start_date=pred_start_date, pred_end_date=pred_end_date,
        day_0=day_0, **input_data
    )
    
    ## STAGE 1 MODEL
    # check what NAs in pred data might be about, get rid of them in safer way
    mr_model_dict, prior_dicts, pred, pred_fe, pred_location_map, age_stand_scaling_factor, level_lambdas = ifr.model.run_model(
        model_data=model_data.copy(),
        pred_data=pred_data.copy(),
        day_0=day_0, day_inflection=day_inflection,
        verbose=verbose,
        **input_data
    )
    
    ## REINFECTION
    # account for escape variant re-infection
    reinfection_inflation_factor, seroprevalence = reinfection.add_repeat_infections(
        input_data['escape_variant_prevalence'].copy(),
        input_data['daily_deaths'].copy(),
        pred.copy(),
        input_data['seroprevalence'].copy(),
        input_data['hierarchy'],
        input_data['gbd_hierarchy'],
        input_data['population'],
        verbose=verbose,
    )

    ## WANING SENSITIVITY ADJUSTMENT
    # account for waning antibody detection

    hospitalized_weights = age_standardization.get_all_age_rate(
        input_data['ihr_age_pattern'].copy(), input_data['sero_age_pattern'].copy(),
        input_data['age_spec_population'].copy()
    )
    sensitivity, seroprevalence = serology.apply_waning_adjustment(
        input_data['sensitivity'].copy(),
        input_data['assay_map'].copy(),
        hospitalized_weights.copy(),
        input_data['seroprevalence'].copy(),
        input_data['daily_deaths'].copy(),
        pred.copy(),
    )
    
    ## SET UP REFIT
    refit_input_data = input_data.copy()
    refit_input_data['seroprevalence'] = seroprevalence
    del seroprevalence
    refit_model_data = ifr.data.create_model_data(day_0=day_0, **refit_input_data)
    refit_pred_data = ifr.data.create_pred_data(
        pred_start_date=pred_start_date, pred_end_date=pred_end_date,
        day_0=day_0, **refit_input_data
    )
    
    ## STAGE 2 MODEL
    # check what NAs in pred data might be about, get rid of them in safer way
    refit_mr_model_dict, refit_prior_dicts, refit_pred, refit_pred_fe, \
    refit_pred_location_map, refit_age_stand_scaling_factor, refit_level_lambdas = ifr.model.run_model(
        model_data=refit_model_data.copy(),
        pred_data=refit_pred_data.dropna().copy(),
        day_0=day_0, day_inflection=day_inflection,
        verbose=verbose,
        **refit_input_data
    )
    refit_pred_unadj = refit_pred.copy()
    
    ## POST
    refit_pred, refit_pred_lr, refit_pred_hr, pct_inf_lr, pct_inf_hr = post.variants_vaccines(
        rate_age_pattern=refit_input_data['ifr_age_pattern'].copy(),
        denom_age_pattern=refit_input_data['sero_age_pattern'].copy(),
        age_spec_population=refit_input_data['age_spec_population'].copy(),
        rate=refit_pred.copy(),
        day_shift=EXPOSURE_TO_DEATH,
        escape_variant_prevalence=refit_input_data['escape_variant_prevalence'].copy(),
        severity_variant_prevalence=refit_input_data['severity_variant_prevalence'].copy(),
        vaccine_coverage=refit_input_data['vaccine_coverage'].copy(),
        population=refit_input_data['population'].copy(),
    )
    
    lr_rr = refit_pred_lr / refit_pred
    hr_rr = refit_pred_hr / refit_pred
    refit_pred = squeeze.squeeze(
        daily=refit_input_data['daily_deaths'].copy(),
        rate=refit_pred.copy(),
        day_shift=EXPOSURE_TO_DEATH,
        population=refit_input_data['population'].copy(),
        reinfection_inflation_factor=(reinfection_inflation_factor
                                      .set_index(['location_id', 'date'])
                                      .loc[:, 'inflation_factor']
                                      .copy()),
        vaccine_coverage=refit_input_data['vaccine_coverage'].copy(),
    )
    refit_pred_lr = lr_rr * refit_pred
    refit_pred_hr = hr_rr * refit_pred
    
    results = RESULTS(
        seroprevalence=input_data['seroprevalence'],
        model_data=model_data,
        mr_model_dict=mr_model_dict,
        pred_location_map=pred_location_map,
        level_lambdas=level_lambdas,
        daily_numerator=input_data['daily_deaths'],
        pred=pred,
        pred_unadj=pred,
        pred_fe=pred_fe,
        pred_lr=None,
        pred_hr=None,
        pct_inf_lr=None,
        pct_inf_hr=None,
        age_stand_scaling_factor=age_stand_scaling_factor,
    )
    refit_results = RESULTS(
        seroprevalence=refit_input_data['seroprevalence'],
        model_data=refit_model_data,
        mr_model_dict=refit_mr_model_dict,
        pred_location_map=refit_pred_location_map,
        level_lambdas=refit_level_lambdas,
        daily_numerator=refit_input_data['daily_deaths'],
        pred=refit_pred,
        pred_unadj=refit_pred_unadj,
        pred_fe=refit_pred_fe,
        pred_lr=refit_pred_lr,
        pred_hr=refit_pred_hr,
        pct_inf_lr=pct_inf_lr,
        pct_inf_hr=pct_inf_hr,
        age_stand_scaling_factor=refit_age_stand_scaling_factor,
    )
    
    nrmse, residuals = ifr.model.get_nrmse(refit_input_data['seroprevalence'].copy(),
                                           refit_input_data['daily_deaths'].copy(),
                                           refit_pred.copy(),
                                           refit_input_data['hierarchy'].copy(),
                                           refit_input_data['population'].copy(),
                                           refit_pred_location_map.copy(),
                                           refit_mr_model_dict.copy(),)

    return {'raw_results': results, 'refit_results': refit_results,
            'reinfection_inflation_factor': reinfection_inflation_factor,
            'sensitivity': sensitivity, 'nrmse': nrmse, 'residuals': residuals,}


def main(inputs_path: str, outputs_path: str):
    with Path(inputs_path).open('rb') as file:
        inputs = pickle.load(file)
        
    outputs = runner(**inputs)
    
    with Path(outputs_path).open('wb') as file:
        pickle.dump({inputs['day_inflection']: outputs}, file)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '6'
    configure_logging_to_terminal(verbose=2)
    
    main(sys.argv[1], sys.argv[2])
