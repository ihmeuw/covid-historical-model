from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


from covid_historical_model.durations.durations import SERO_TO_DEATH, EXPOSURE_TO_SEROPOSITIVE
from covid_historical_model.etl import model_inputs


def load_seroprevalence_sub_vacccinated(model_inputs_root: Path, vaccinated: pd.Series,
                                        verbose: bool = True) -> pd.DataFrame:
    seroprevalence = model_inputs.seroprevalence(model_inputs_root, verbose=verbose)
    
    population = model_inputs.population(model_inputs_root)
    vaccinated /= population
    
    vaccinated = vaccinated.reset_index()
    # assume this is date of seroconversion?
    # vaccinated['date'] += pd.Timedelta(days=EXPOSURE_TO_SEROPOSITIVE)
    vaccinated = vaccinated.set_index(['location_id', 'date'])
    
    if verbose:
        logger.info('Removing effectively vaccinated from reported seroprevalence.')
    seroprevalence = remove_vaccinated(seroprevalence, vaccinated,)
    
    return seroprevalence


def remove_vaccinated(seroprevalence: pd.DataFrame,
                      vaccinated: pd.Series,) -> pd.DataFrame:
    ## remove vaccinated based on end date? otherwise use commented out bits here
    # seroprevalence = seroprevalence.rename(columns={'date':'end_date'})
    # seroprevalence['n_midpoint_days'] = (seroprevalence['end_date'] - seroprevalence['start_date']).dt.days / 2
    # seroprevalence['n_midpoint_days'] = seroprevalence['n_midpoint_days'].astype(int)
    # seroprevalence['date'] = seroprevalence.apply(lambda x: x['end_date'] - pd.Timedelta(days=x['n_midpoint_days']), axis=1)
    seroprevalence = seroprevalence.merge(vaccinated.reset_index(), how='left')
    # del seroprevalence['date']
    # del seroprevalence['n_midpoint_days']
    # seroprevalence = seroprevalence.rename(columns={'end_date':'date'})
    seroprevalence['vaccinated'] = seroprevalence['vaccinated'].fillna(0)
    
    seroprevalence.loc[seroprevalence['test_target'] != 'spike', 'vaccinated'] = 0
    
    seroprevalence = seroprevalence.rename(columns={'seroprevalence':'reported_seroprevalence'})
    
    seroprevalence['seroprevalence'] = 1 - (1 - seroprevalence['reported_seroprevalence']) / (1 - seroprevalence['vaccinated'])
    
    del seroprevalence['vaccinated']
    
    return seroprevalence


def apply_waning_adjustment(model_inputs_root: Path,
                            hospitalized_weights: pd.Series,
                            seroprevalence: pd.DataFrame,
                            daily_deaths: pd.Series,
                            pred_ifr: pd.Series,
                            verbose: bool = True,) -> pd.DataFrame:
    sensitivity = model_inputs.assay_sensitivity(model_inputs_root)

    assays = ['N-Abbott',  # IgG
              'S-Roche', 'N-Roche',  # Ig
              'S-Ortho Ig', 'S-Ortho IgG', # Ig/IgG
              'S-DiaSorin',  # IgG
              'S-EuroImmun',]  # IgG
    increasing = ['S-Ortho Ig']
    
    data_assays = sensitivity['assay'].unique().tolist()
    excluded_data_assays = [da for da in data_assays if da not in assays]
    if verbose and excluded_data_assays:
        logger.warning(f"Excluding the following assays found in sensitivity data: {', '.join(excluded_data_assays)}")
    if any([a not in data_assays for a in assays]):
        raise ValueError('Assay mis-labelled.')
    sensitivity = sensitivity.loc[sensitivity['assay'].isin(assays)]
    
    source_assays = sensitivity[['source', 'assay']].drop_duplicates().values.tolist()
    
    sensitivity = pd.concat(
        [
            fit_hospital_weighted_sensitivity_decay(
                sensitivity.loc[(sensitivity['source'] == source) & (sensitivity['assay'] == assay)].copy(),
                assay in increasing,
                hospitalized_weights.copy()
            )
            for source, assay in source_assays]
    ).set_index(['assay', 'location_id', 't']).sort_index()
    
    seroprevalence = seroprevalence.loc[seroprevalence['is_outlier'] == 0]
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## DO THIS DIFFERENTLY...
    test_matching = pd.read_csv('/'.join(__file__.split('/')[:-2]) + '/tests.csv',
                                encoding='latin1')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    
    seroprevalence = seroprevalence.merge(test_matching, how='left')
    missing_match = seroprevalence['assay_match'].isnull()
    is_N = seroprevalence['test_target'] == 'nucleocapsid'
    is_S = seroprevalence['test_target'] == 'spike'
    is_other = ~(is_N | is_S)
    seroprevalence.loc[missing_match & is_N, 'assay_match'] = 'N-Roche, N-Abbott'
    seroprevalence.loc[missing_match & is_S, 'assay_match'] = 'S-Roche, S-Ortho Ig, S-Ortho IgG, S-DiaSorin, S-EuroImmun'
    seroprevalence.loc[missing_match & is_other, 'assay_match'] = 'N-Roche, ' \
                                                                  'N-Abbott, ' \
                                                                  'S-Roche, S-Ortho Ig, ' \
                                                                  'S-Ortho IgG, S-DiaSorin, S-EuroImmun' 

    assay_combinations = seroprevalence['assay_match'].unique().tolist()

    sensitivity_list = []
    seroprevalence_list = []
    for assay_combination in assay_combinations:
        ac_sensitivity = (sensitivity
                             .loc[assay_combination.split(', ')]
                             .reset_index()
                             .groupby(['location_id', 't'])['sensitivity'].mean())
        ac_seroprevalence = (seroprevalence
                             .loc[seroprevalence['assay_match'] == assay_combination].copy())
        ac_seroprevalence = waning_adjustment(
            pred_ifr.copy(),
            daily_deaths.copy(),
            ac_sensitivity.copy(),
            ac_seroprevalence.copy()
        )
        
        ac_sensitivity = (ac_sensitivity
                          .loc[ac_seroprevalence['location_id'].unique().tolist()]
                          .reset_index())
        ac_sensitivity['assay'] = assay_combination
        sensitivity_list.append(ac_sensitivity)
        
        ac_seroprevalence['is_outlier'] = 0
        ac_seroprevalence['assay'] = assay_combination
        seroprevalence_list.append(ac_seroprevalence)
    sensitivity = pd.concat(sensitivity_list)
    seroprevalence = pd.concat(seroprevalence_list)
    
    return sensitivity, seroprevalence


def fit_sensitivity_decay(t: np.array, sensitivity: np.array, increasing: bool, t_N: int = 720) -> pd.DataFrame:
    def sigmoid(x, x0, k):
        y = 1 / (1 + np.exp(-k * (x-x0)))
        return y
    
    if increasing:
        bounds = ([-np.inf, 1e-4], [np.inf, 0.5])
    else:
        bounds = ([-np.inf, -0.5], [np.inf, -1e-4])
    popt, pcov = curve_fit(sigmoid,
                           t, sensitivity,
                           method='dogbox',
                           bounds=bounds, max_nfev=1000)
    
    t_pred = np.arange(0, t_N + 1)
    sensitivity_pred = sigmoid(t_pred, *popt)
        
    return pd.DataFrame({'t': t_pred, 'sensitivity': sensitivity_pred})


def fit_hospital_weighted_sensitivity_decay(sensitivity: pd.DataFrame, increasing: bool,
                                            hospitalized_weights: pd.Series,) -> pd.DataFrame:
    assay = sensitivity['assay'].unique().item()
    hosp_sensitivity = sensitivity.loc[sensitivity['hospitalization_status'] == 'Hospitalized']
    nonhosp_sensitivity = sensitivity.loc[sensitivity['hospitalization_status'] == 'Non-hospitalized']
    hosp_sensitivity = fit_sensitivity_decay(hosp_sensitivity['t'].values,
                                             hosp_sensitivity['sensitivity'].values,
                                             increasing)
    nonhosp_sensitivity = fit_sensitivity_decay(nonhosp_sensitivity['t'].values,
                                                nonhosp_sensitivity['sensitivity'].values,
                                                increasing)
    sensitivity = (hosp_sensitivity
                   .rename(columns={'sensitivity':'hosp_sensitivity'})
                   .merge(nonhosp_sensitivity
                          .rename(columns={'sensitivity':'nonhosp_sensitivity'})))
    sensitivity['key'] = 0
    hospitalized_weights = hospitalized_weights.rename('hospitalized_weights')
    hospitalized_weights = hospitalized_weights.reset_index()
    hospitalized_weights['key'] = 0
    sensitivity = sensitivity.merge(hospitalized_weights, on='key', how='outer')
    sensitivity['sensitivity'] = (sensitivity['hosp_sensitivity'] * sensitivity['hospitalized_weights']) + \
                                 (sensitivity['nonhosp_sensitivity'] * (1 - sensitivity['hospitalized_weights']))
    sensitivity = sensitivity.reset_index()
    sensitivity['assay'] = assay
    
    return sensitivity.loc[:, ['location_id', 'assay', 't', 'sensitivity', 'hosp_sensitivity', 'nonhosp_sensitivity']]


def calulate_waning_factor(infections: pd.DataFrame, sensitivity: pd.DataFrame,
                           sero_date: pd.Timestamp) -> float:
    infections['t'] = (sero_date - infections['date']).dt.days
    infections = infections.merge(sensitivity.reset_index())
    waning_factor = infections['infections'].sum() / (infections['infections'] * infections['sensitivity']).sum()
    waning_factor = max(1, waning_factor)

    return waning_factor
    
    
def location_waning_adjustment(infections: pd.DataFrame, sensitivity: pd.DataFrame,
                               seroprevalence: pd.DataFrame) -> pd.DataFrame:
    adj_seroprevalence = []
    for i, (sero_data_id, sero_date, sero_value) in enumerate(zip(seroprevalence['data_id'],
                                                                  seroprevalence['date'],
                                                                  seroprevalence['seroprevalence'])):
        waning_factor = calulate_waning_factor(infections, sensitivity, sero_date)
        adj_seroprevalence.append(pd.DataFrame({
            'data_id': sero_data_id,
            'date': sero_date,
            'seroprevalence': sero_value * waning_factor
        }, index=[i]))
    adj_seroprevalence = pd.concat(adj_seroprevalence)
    
    return adj_seroprevalence


def waning_adjustment(pred_ifr: pd.Series, daily_deaths: pd.Series, sensitivity: pd.DataFrame,
                      seroprevalence: pd.DataFrame) -> pd.DataFrame:
    infections = ((daily_deaths / pred_ifr)
                  .dropna()
                  .rename('infections')
                  .reset_index()
                  .set_index('location_id'))
    infections['date'] -= pd.Timedelta(days=SERO_TO_DEATH)
    
    # determine waning adjustment based on midpoint of survey
    orig_date = seroprevalence[['data_id', 'date']].copy()
    seroprevalence['n_midpoint_days'] = (seroprevalence['date'] - seroprevalence['start_date']).dt.days / 2
    seroprevalence['n_midpoint_days'] = seroprevalence['n_midpoint_days'].astype(int)
    seroprevalence['date'] = seroprevalence.apply(lambda x: x['date'] - pd.Timedelta(days=x['n_midpoint_days']), axis=1)
    del seroprevalence['n_midpoint_days']
    
    seroprevalence_list = []
    location_ids = seroprevalence['location_id'].unique().tolist()
    location_ids = [location_id for location_id in location_ids if location_id in infections.reset_index()['location_id'].to_list()]
    for location_id in location_ids:
        _sero = location_waning_adjustment(infections.loc[location_id],
                                           sensitivity.loc[location_id],
                                           (seroprevalence
                                            .loc[seroprevalence['location_id'] == location_id,
                                                 ['data_id', 'date', 'seroprevalence']
                                                ].reset_index(drop=True)))
        _sero['location_id'] = location_id
        seroprevalence_list.append(_sero)
    seroprevalence = pd.concat(seroprevalence_list).reset_index(drop=True)
    del seroprevalence['date']
    seroprevalence = seroprevalence.merge(orig_date)
    
    return seroprevalence
