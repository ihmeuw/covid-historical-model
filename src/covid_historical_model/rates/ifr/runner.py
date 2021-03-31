from pathlib import Path
from typing import Dict
import itertools
from collections import namedtuple
from loguru import logger

import pandas as pd

from covid_historical_model.rates import ifr
from covid_historical_model.rates import ihr
from covid_historical_model.rates import reinfection
from covid_historical_model.rates import serology
from covid_historical_model.rates import age_standardization
from covid_historical_model.rates import post

RESULTS = namedtuple('Results', 'seroprevalence model_data mr_model_dict pred_location_map pred pred_fe pred_lr pred_hr')


def runner(model_inputs_root: Path, em_path: Path, age_pattern_root: Path, variant_scaleup_root: Path,
           orig_seroprevalence: pd.DataFrame, vaccine_coverage: pd.DataFrame,
           day_inflection: str,
           day_0: str = '2020-03-15',
           pred_start_date: str = '2020-01-01',
           pred_end_date: str = '2021-12-31',
           verbose: bool = True,) -> Dict:
    day_inflection = pd.Timestamp(day_inflection)
    day_0 = pd.Timestamp(day_0)
    pred_start_date = pd.Timestamp(pred_start_date)
    pred_end_date = pd.Timestamp(pred_end_date)

    input_data = ifr.data.load_input_data(model_inputs_root, em_path, age_pattern_root, variant_scaleup_root,
                                          orig_seroprevalence, vaccine_coverage, verbose=verbose)
    model_data = ifr.data.create_model_data(day_0=day_0, **input_data)
    pred_data = ifr.data.create_pred_data(
        pred_start_date=pred_start_date, pred_end_date=pred_end_date,
        day_0=day_0, **input_data
    )
    
    # check what NAs in pred data might be about, get rid of them in safer way
    mr_model_dict, prior_dicts, pred, pred_fe, pred_location_map = ifr.model.run_model(
        model_data=model_data.copy(),
        pred_data=pred_data.copy(),
        day_0=day_0, day_inflection=day_inflection,
        verbose=verbose,
        **input_data
    )
    
    # account for escape variant re-infection
    reinfection_inflation_factor, seroprevalence = reinfection.add_repeat_infections(
        input_data['variant_prevalence'].copy(),
        input_data['daily_deaths'].copy(),
        pred.copy(),
        orig_seroprevalence.copy(),
        input_data['hierarchy'],
        input_data['population'],
        verbose=verbose,
    )

    # account for waning antibody detection
    ihr_age_pattern = ihr.data.load_input_data(model_inputs_root, age_pattern_root, variant_scaleup_root,
                                               seroprevalence, vaccine_coverage, verbose=verbose)['ihr_age_pattern']
    hospitalized_weights = age_standardization.get_all_age_rate(
        ihr_age_pattern, input_data['sero_age_pattern'],
        input_data['age_spec_population']
    )
    sensitivity, seroprevalence = serology.apply_waning_adjustment(
        model_inputs_root,
        hospitalized_weights.copy(),
        seroprevalence.copy(),
        input_data['daily_deaths'].copy(),
        pred.copy(),
    )
    
    refit_input_data = input_data.copy()
    refit_input_data['seroprevalence'] = seroprevalence
    refit_model_data = ifr.data.create_model_data(day_0=day_0, **refit_input_data)
    refit_pred_data = ifr.data.create_pred_data(
        pred_start_date=pred_start_date, pred_end_date=pred_end_date,
        day_0=day_0, **refit_input_data
    )

    # check what NAs in pred data might be about, get rid of them in safer way
    refit_mr_model_dict, refit_prior_dicts, refit_pred, refit_pred_fe, refit_pred_location_map = ifr.model.run_model(
        model_data=refit_model_data.copy(),
        pred_data=refit_pred_data.dropna().copy(),
        day_0=day_0, day_inflection=day_inflection,
        verbose=verbose,
        **refit_input_data
    )
    
    pred, pred_lr, pred_hr = post.variants_vaccines(
        rate_age_pattern=input_data['ifr_age_pattern'].copy(),
        denom_age_pattern=input_data['sero_age_pattern'].copy(),
        age_spec_population=input_data['age_spec_population'].copy(),
        numerator=input_data['daily_deaths'].copy(),
        rate=pred.copy(),
        variant_prevalence=input_data['variant_prevalence'].copy(),
        vaccine_coverage=input_data['vaccine_coverage'].copy(),
        population=input_data['population'].copy(),
    )
    refit_pred, refit_pred_lr, refit_pred_hr = post.variants_vaccines(
        rate_age_pattern=refit_input_data['ifr_age_pattern'].copy(),
        denom_age_pattern=refit_input_data['sero_age_pattern'].copy(),
        age_spec_population=refit_input_data['age_spec_population'].copy(),
        numerator=refit_input_data['daily_deaths'].copy(),
        rate=refit_pred.copy(),
        variant_prevalence=refit_input_data['variant_prevalence'].copy(),
        vaccine_coverage=refit_input_data['vaccine_coverage'].copy(),
        population=refit_input_data['population'].copy(),
    )
    
    results = RESULTS(
        seroprevalence=orig_seroprevalence,
        model_data=model_data,
        mr_model_dict=mr_model_dict,
        pred_location_map=pred_location_map,
        pred=pred,
        pred_fe=pred_fe,
        pred_lr=pred_lr,
        pred_hr=pred_hr,
    )
    refit_results = RESULTS(
        seroprevalence=seroprevalence,
        model_data=refit_model_data,
        mr_model_dict=refit_mr_model_dict,
        pred_location_map=refit_pred_location_map,
        pred=refit_pred,
        pred_fe=refit_pred_fe,
        pred_lr=refit_pred_lr,
        pred_hr=refit_pred_hr,
    )
    
    nrmse = ifr.model.get_nrmse(seroprevalence.copy(),
                                refit_input_data['daily_deaths'].copy(),
                                refit_pred.copy(),
                                refit_input_data['population'].copy(),
                                refit_pred_location_map.copy(),
                                refit_mr_model_dict.copy(),)

    return {'raw_results': results, 'refit_results': refit_results,
            'reinfection_inflation_factor': reinfection_inflation_factor,
            'sensitivity': sensitivity, 'nrmse': nrmse,}
