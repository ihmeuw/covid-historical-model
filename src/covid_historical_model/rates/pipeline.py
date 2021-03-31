from typing import List, Tuple, Dict
from pathlib import Path
from loguru import logger

import pandas as pd

from covid_historical_model.etl import estimates
from covid_historical_model.rates import serology
from covid_historical_model.rates import ifr
from covid_historical_model.rates import idr
from covid_historical_model.rates import ihr


def pipeline(model_inputs_root: Path, em_path: Path,
             vaccine_coverage_root: Path, variant_scaleup_root: Path,
             age_pattern_root: Path, testing_root: Path,
             day_inflection_list: List[str] = ['2020-05-01', '2020-06-01', '2020-07-01',
                                               '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01'],
             verbose: bool = True,) -> Tuple:
    vaccine_coverage = estimates.vaccine_coverage(vaccine_coverage_root)
    seroprevalence = serology.load_seroprevalence_sub_vacccinated(
        model_inputs_root, vaccine_coverage['cumulative_all_effective'].rename('vaccinated')
    )
    
    full_ifr_results = {}
    for day_inflection in day_inflection_list:
        if verbose:
            logger.info('\n*************************************\n'
                        f'IFR ESTIMATION -- testing inflection point at {day_inflection}\n'
                        '*************************************')
        full_ifr_results.update({day_inflection: ifr.runner.runner(model_inputs_root, em_path, age_pattern_root,
                                                                   variant_scaleup_root,
                                                                   seroprevalence.copy(), vaccine_coverage.copy(),
                                                                   day_inflection,
                                                                   verbose=verbose)})
    
    if verbose:
        logger.info('\n*************************************\n'
                    'IFR ESTIMATION -- determining best models and compiling adjusted seroprevalence\n'
                    '*************************************')
    ifr_results, seroprevalence, sensitivity, reinfection_inflation_factor = extract_ifr_results(full_ifr_results)

    if verbose:
        logger.info('\n*************************************\n'
                    'IDR ESTIMATION\n'
                    '*************************************')
    idr_results = idr.runner.runner(model_inputs_root, testing_root,
                                    adj_seroprevalence.copy(),
                                    ifr_results.pred.copy(),
                                    verbose=verbose)
    
    if verbose:
        logger.info('\n*************************************\n'
                    'IHR ESTIMATION\n'
                    '*************************************')
    ihr_results = ihr.runner.runner(model_inputs_root, age_pattern_root, variant_scaleup_root,
                                    adj_seroprevalence.copy(), vaccine_coverage.copy(), verbose=verbose)
    
    return seroprevalence, sensitivity, reinfection_inflation_factor, ifr_results, idr_results, ihr_results


def extract_ifr_results(full_ifr_results: Dict) -> Tuple:
    nrmse = []
    for day_inflection, di_results in full_ifr_results.items():
        nrmse.append(pd.concat([di_results['nrmse'],
                                pd.DataFrame({'day_inflection':day_inflection},
                                             index=di_results['nrmse'].index)],
                               axis=1))
    nrmse = pd.concat(nrmse).reset_index()
    best_models = (nrmse
                   .groupby('location_id')
                   .apply(lambda x: x.sort_values('nrmse')['day_inflection'].values[0])
                   .rename('day_inflection')
                   .reset_index())

    sensitivity = pd.DataFrame({'location_id':[]})
    reinfection_inflation_factor = pd.DataFrame()
    ifr_seroprevalence = []
    ifr_model_data = []
    ifr_mr_model_dict = {}
    ifr_pred_location_map = {}
    ifr_pred = []
    ifr_pred_fe = []
    ifr_pred_lr = []
    ifr_pred_hr = []
    for location_id, day_inflection in zip(best_models['location_id'], best_models['day_inflection']):
        loc_seroprevalence = full_ifr_results[day_inflection]['refit_results'].seroprevalence
        loc_seroprevalence = loc_seroprevalence.loc[loc_seroprevalence['location_id'] == location_id]
        ifr_seroprevalence.append(loc_seroprevalence)
        
        loc_rif = full_ifr_results[day_inflection]['reinfection_inflation_factor']
        loc_rif = loc_rif.loc[loc_rif['location_id'] == location_id]
        reinfection_inflation_factor = pd.concat([reinfection_inflation_factor, loc_rif])

        loc_model_data = full_ifr_results[day_inflection]['refit_results'].model_data
        loc_model_data = loc_model_data.loc[loc_model_data['location_id'] == location_id]
        ifr_model_data.append(loc_model_data)
        
        try:  # extract pred map and model object in this chunk
            loc_model_location = full_ifr_results[day_inflection]['refit_results'].pred_location_map[location_id]
            ifr_pred_location_map.update({location_id: loc_model_location})
            if loc_model_location not in list(ifr_mr_model_dict.keys()):
                loc_mr_model = full_ifr_results[day_inflection]['refit_results'].mr_model_dict
                loc_mr_model = loc_mr_model[loc_model_location]
                ifr_mr_model_dict.update({loc_model_location: loc_mr_model})
            if loc_model_location not in sensitivity['location_id'].to_list():
                loc_sensitivity = full_ifr_results[day_inflection]['sensitivity']
                loc_sensitivity = loc_sensitivity.loc[loc_sensitivity['location_id'] == loc_model_location]
                sensitivity = pd.concat([sensitivity, loc_sensitivity])
        except KeyError:
            pass
        
        try:
            loc_pred = full_ifr_results[day_inflection]['refit_results'].pred.loc[[location_id]]
            ifr_pred.append(loc_pred)
        except KeyError:
            pass

        try:
            loc_pred_fe = full_ifr_results[day_inflection]['refit_results'].pred_fe.loc[[location_id]]
            ifr_pred_fe.append(loc_pred_fe)
        except KeyError:
            pass

        try:
            loc_pred_lr = full_ifr_results[day_inflection]['refit_results'].pred_lr.loc[[location_id]]
            ifr_pred_lr.append(loc_pred_lr)
        except KeyError:
            pass

        try:
            loc_pred_hr = full_ifr_results[day_inflection]['refit_results'].pred_hr.loc[[location_id]]
            ifr_pred_hr.append(loc_pred_hr)
        except KeyError:
            pass
    sensitivity = sensitivity.reset_index(drop=True)
    reinfection_inflation_factor = reinfection_inflation_factor.reset_index(drop=True)
    ifr_results = ifr.runner.RESULTS(
        seroprevalence=pd.concat(ifr_seroprevalence).reset_index(drop=True),
        model_data=pd.concat(ifr_model_data).reset_index(drop=True),
        mr_model_dict=ifr_mr_model_dict,
        pred_location_map=ifr_pred_location_map,
        pred=pd.concat(ifr_pred),
        pred_fe=pd.concat(ifr_pred_fe),
        pred_lr=pd.concat(ifr_pred_lr),
        pred_hr=pd.concat(ifr_pred_hr),
    )
    for day_inflection, di_results in full_ifr_results.items():
        nrmse.append(pd.concat([di_results['nrmse'],
                                pd.DataFrame({'day_inflection':day_inflection},
                                             index=di_results['nrmse'].index)],
                               axis=1))
    nrmse = pd.concat(nrmse).reset_index()
    best_models = (nrmse
                   .groupby('location_id')
                   .apply(lambda x: x.sort_values('nrmse')['day_inflection'].values[0])
                   .rename('day_inflection')
                   .reset_index())

    sensitivity = pd.DataFrame({'location_id':[]})
    reinfection_inflation_factor = pd.DataFrame()
    ifr_seroprevalence = []
    ifr_model_data = []
    ifr_mr_model_dict = {}
    ifr_pred_location_map = {}
    ifr_pred = []
    ifr_pred_fe = []
    ifr_pred_lr = []
    ifr_pred_hr = []
    for location_id, day_inflection in zip(best_models['location_id'], best_models['day_inflection']):
        loc_seroprevalence = full_ifr_results[day_inflection]['refit_results'].seroprevalence
        loc_seroprevalence = loc_seroprevalence.loc[loc_seroprevalence['location_id'] == location_id]
        ifr_seroprevalence.append(loc_seroprevalence)
        
        loc_rif = full_ifr_results[day_inflection]['reinfection_inflation_factor']
        loc_rif = loc_rif.loc[loc_rif['location_id'] == location_id]
        reinfection_inflation_factor = pd.concat([reinfection_inflation_factor, loc_rif])

        loc_model_data = full_ifr_results[day_inflection]['refit_results'].model_data
        loc_model_data = loc_model_data.loc[loc_model_data['location_id'] == location_id]
        ifr_model_data.append(loc_model_data)
        
        try:  # extract pred map and model object in this chunk
            loc_model_location = full_ifr_results[day_inflection]['refit_results'].pred_location_map[location_id]
            ifr_pred_location_map.update({location_id: loc_model_location})
            if loc_model_location not in list(ifr_mr_model_dict.keys()):
                loc_mr_model = full_ifr_results[day_inflection]['refit_results'].mr_model_dict
                loc_mr_model = loc_mr_model[loc_model_location]
                ifr_mr_model_dict.update({loc_model_location: loc_mr_model})
            if loc_model_location not in sensitivity['location_id'].to_list():
                loc_sensitivity = full_ifr_results[day_inflection]['sensitivity']
                loc_sensitivity = loc_sensitivity.loc[loc_sensitivity['location_id'] == loc_model_location]
                sensitivity = pd.concat([sensitivity, loc_sensitivity])
        except KeyError:
            pass
        
        try:
            loc_pred = full_ifr_results[day_inflection]['refit_results'].pred.loc[[location_id]]
            ifr_pred.append(loc_pred)
        except KeyError:
            pass

        try:
            loc_pred_fe = full_ifr_results[day_inflection]['refit_results'].pred_fe.loc[[location_id]]
            ifr_pred_fe.append(loc_pred_fe)
        except KeyError:
            pass

        try:
            loc_pred_lr = full_ifr_results[day_inflection]['refit_results'].pred_lr.loc[[location_id]]
            ifr_pred_lr.append(loc_pred_lr)
        except KeyError:
            pass

        try:
            loc_pred_hr = full_ifr_results[day_inflection]['refit_results'].pred_hr.loc[[location_id]]
            ifr_pred_hr.append(loc_pred_hr)
        except KeyError:
            pass
    sensitivity = sensitivity.reset_index(drop=True)
    reinfection_inflation_factor = reinfection_inflation_factor.reset_index(drop=True)
    ifr_results = ifr.runner.RESULTS(
        seroprevalence=pd.concat(ifr_seroprevalence).reset_index(drop=True),
        model_data=pd.concat(ifr_model_data).reset_index(drop=True),
        mr_model_dict=ifr_mr_model_dict,
        pred_location_map=ifr_pred_location_map,
        pred=pd.concat(ifr_pred),
        pred_fe=pd.concat(ifr_pred_fe),
        pred_lr=pd.concat(ifr_pred_lr),
        pred_hr=pd.concat(ifr_pred_hr),
    )
    seroprevalence = ifr_results.seroprevalence.copy()

    return ifr_results, seroprevalence, sensitivity, reinfection_inflation_factor
