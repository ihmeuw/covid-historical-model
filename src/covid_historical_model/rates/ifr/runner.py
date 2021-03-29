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

RESULTS = namedtuple('Results', 'seroprevalence model_data mr_model_dict pred_location_map pred pred_fe pred_lr pred_hr')


def runner(model_inputs_root: Path, age_pattern_root: Path, variant_scaleup_root: Path,
           orig_seroprevalence: pd.DataFrame,
           day_inflection: str,
           day_0: str = '2020-03-15',
           pred_start_date: str = '2020-01-01',
           pred_end_date: str = '2021-12-31',
           verbose: bool = True,) -> Dict:
    day_inflection = pd.Timestamp(day_inflection)
    day_0 = pd.Timestamp(day_0)
    pred_start_date = pd.Timestamp(pred_start_date)
    pred_end_date = pd.Timestamp(pred_end_date)

    input_data = ifr.data.load_input_data(model_inputs_root, age_pattern_root,
                                          orig_seroprevalence, verbose=verbose)
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
    seroprevalence = reinfection.add_repeat_infections(
        variant_scaleup_root,
        input_data['daily_deaths'].copy(),
        pred.copy(),
        orig_seroprevalence.copy(),
    )

    # account for waning antibody detection
    ihr_age_pattern = ihr.data.load_input_data(model_inputs_root, age_pattern_root,
                                               seroprevalence, verbose=verbose)['ihr_age_pattern']
    hospitalized_weights = age_standardization.get_all_age_rate(
        ihr_age_pattern, input_data['sero_age_pattern'],
        input_data['age_spec_population']
    )
    seroprevalence = serology.apply_waning_adjustment(
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
    
    low_risk_rr, high_risk_rr = age_standardization.get_risk_group_rr(
        input_data['ifr_age_pattern'].copy(),
        input_data['sero_age_pattern'].copy(),
        input_data['age_spec_population'].copy(),
    )
    pred_lr = (pred * low_risk_rr).rename('pred_ifr_lr')
    pred_hr = (pred * high_risk_rr).rename('pred_ifr_hr')
    refit_pred_lr = (refit_pred * low_risk_rr).rename('pred_ifr_lr')
    refit_pred_hr = (refit_pred * high_risk_rr).rename('pred_ifr_hr')
    
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

    return {'raw_results': results, 'refit_results': refit_results, 'nrmse': nrmse}
