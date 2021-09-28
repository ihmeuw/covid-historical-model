import sys
from typing import Dict
from pathlib import Path
from loguru import logger
from collections import namedtuple
from datetime import datetime
import functools
import multiprocessing
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns

from covid_historical_model.etl import model_inputs
from covid_historical_model.utils.misc import text_wrap, get_random_state
from covid_historical_model.utils.math import logit, expit, scale_to_bounds
from covid_historical_model.cluster import CONTROLLER_MP_THREADS, OMP_NUM_THREADS
from covid_historical_model.mrbrt import mrbrt

VAX_SERO_PROB = 0.9
SEROREV_LB = 0.

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# should have module for these that is more robust to additions
ASSAYS = ['N-Abbott',  # IgG
          'S-Roche', 'N-Roche',  # Ig
          'S-Ortho Ig', 'S-Ortho IgG', # Ig/IgG
          'S-DiaSorin',  # IgG
          'S-EuroImmun',  # IgG
          'S-Oxford',]  # IgG
INCREASING = ['S-Ortho Ig', 'S-Roche']
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

PLOT_DATE_LOCATOR = mdates.AutoDateLocator(maxticks=10)
PLOT_DATE_FORMATTER = mdates.ConciseDateFormatter(PLOT_DATE_LOCATOR, show_offset=False)

PLOT_C_LIST  = ['cornflowerblue', 'lightcoral'   , 'mediumseagreen', 'plum'        ,
                'navajowhite'   , 'paleturquoise', 'hotpink'       , 'peru'        ,
                'palegreen'     , 'lightgrey'    ,]
PLOT_EC_LIST = ['mediumblue'    , 'darkred'      , 'darkgreen'     , 'rebeccapurple',
                'orange'        , 'teal'         , 'deeppink'      , 'saddlebrown'  ,
                'darkseagreen'  , 'grey'         ,]

PLOT_INF_C = 'darkgrey'

PLOT_START_DATE = pd.Timestamp('2020-03-01')
PLOT_END_DATE = pd.Timestamp(str(datetime.today().date()))


def bootstrap(sample: pd.DataFrame,):
    n = sample['n'].unique().item()
    random_state = get_random_state(f'bootstrap_{n}')
    rows = random_state.choice(sample.index, size=len(sample), replace=True)
    # rows = np.random.choice(sample.index, size=len(sample), replace=True)
    bootstraped_samples = []
    for row in rows:
        bootstraped_samples.append(sample.loc[[row]])
        
    return pd.concat(bootstraped_samples).reset_index(drop=True).drop('n', axis=1)


def sample_seroprevalence(seroprevalence: pd.DataFrame, n_samples: int,
                          correlate_samples: bool, bootstrap_samples: bool,
                          min_samples: int = 10,
                          floor: float = 1e-5, logit_se_cap: float = 1.,
                          verbose: bool = True):
    logit_se_from_ci = lambda x: (logit(x['seroprevalence_upper']) - logit(x['seroprevalence_lower'])) / 3.92
    logit_se_from_ss = lambda x: np.sqrt((x['seroprevalence'] * (1 - x['seroprevalence'])) / x['sample_size']) / \
                                 (x['seroprevalence'] * (1.0 - x['seroprevalence']))
    
    series_vars = ['location_id', 'is_outlier', 'survey_series', 'date']
    seroprevalence = seroprevalence.sort_values(series_vars).reset_index(drop=True)
    
    if n_samples >= min_samples:
        if verbose:
            logger.info(f'Producing {n_samples} seroprevalence samples.')
        if (seroprevalence['seroprevalence'] < seroprevalence['seroprevalence_lower']).any():
            mean_sub_low = seroprevalence['seroprevalence'] < seroprevalence['seroprevalence_lower']
            raise ValueError(f'Mean seroprevalence below lower:\n{seroprevalence[mean_sub_low]}')
        if (seroprevalence['seroprevalence'] > seroprevalence['seroprevalence_upper']).any():
            high_sub_mean = seroprevalence['seroprevalence'] > seroprevalence['seroprevalence_upper']
            raise ValueError(f'Mean seroprevalence above upper:\n{seroprevalence[high_sub_mean]}')
            
        summary_vars = ['seroprevalence', 'seroprevalence_lower', 'seroprevalence_upper']
        seroprevalence[summary_vars] = seroprevalence[summary_vars].clip(floor, 1 - floor)

        logit_mean = logit(seroprevalence['seroprevalence'].copy())
        logit_se = logit_se_from_ci(seroprevalence.copy())
        logit_se = logit_se.fillna(logit_se_from_ss(seroprevalence.copy()))
        logit_se = logit_se.fillna(logit_se_cap)
        logit_se = logit_se.clip(0, logit_se_cap)
        logit_samples = np.random.normal(loc=logit_mean.to_frame().values,
                                         scale=logit_se.to_frame().values,
                                         size=(len(seroprevalence), n_samples),)
        samples = expit(logit_samples)
        
        ## CANNOT DO THIS, MOVES SOME ABOVE 1
        # # re-center around original mean
        # samples *= seroprevalence[['seroprevalence']].values / samples.mean(axis=1, keepdims=True)
        if correlate_samples:
            logger.info('Correlating seroprevalence samples.')
            series_data = (seroprevalence[[sv for sv in series_vars if sv not in ['survey_series', 'date']]]
                           .drop_duplicates()
                           .reset_index(drop=True))
            series_data['series'] = series_data.index
            series_data = seroprevalence.merge(series_data).reset_index(drop=True)
            series_idx_list = [series_data.loc[series_data['series'] == series].index.to_list()
                               for series in range(series_data['series'].max() + 1)]
            sorted_samples = []
            for series_idx in series_idx_list:
                series_samples = samples[series_idx, :].copy()
                series_draw_idx = series_samples[0].argsort().argsort()
                series_samples = np.sort(series_samples, axis=1)[:, series_draw_idx]
                sorted_samples.append(series_samples)
            samples = np.vstack(sorted_samples)
            ## THIS SORTS THE WHOLE SET
            # samples = np.sort(samples, axis=1)

        seroprevalence = seroprevalence.drop(['seroprevalence', 'seroprevalence_lower', 'seroprevalence_upper', 'sample_size'],
                                             axis=1)
        sample_list = []
        for n, sample in enumerate(samples.T):
            _sample = seroprevalence.copy()
            _sample['seroprevalence'] = sample
            _sample['n'] = n
            sample_list.append(_sample.reset_index(drop=True))

    elif n_samples > 1:
        raise ValueError(f'If sampling, need at least {min_samples}.')
    else:
        if verbose:
            logger.info('Just using mean seroprevalence.')
            
        seroprevalence['seroprevalence'] = seroprevalence['seroprevalence'].clip(floor, 1 - floor)
            
        seroprevalence = seroprevalence.drop(['seroprevalence_lower', 'seroprevalence_upper', 'sample_size'],
                                             axis=1)
        
        seroprevalence['n'] = 0
        
        sample_list = [seroprevalence.reset_index(drop=True)]

    if bootstrap_samples:
        if n_samples < min_samples:
            raise ValueError('Not set up to bootstrap means only.')
        with multiprocessing.Pool(CONTROLLER_MP_THREADS) as p:
            bootstrap_list = list(tqdm(p.imap(bootstrap, sample_list), total=n_samples, file=sys.stdout))
    else:
        bootstrap_list = sample_list
    
    return bootstrap_list


def load_seroprevalence_sub_vacccinated(model_inputs_root: Path, vaccinated: pd.Series,
                                        n_samples: int, correlate_samples: bool, bootstrap: bool,
                                        verbose: bool = True,) -> pd.DataFrame:
    seroprevalence = model_inputs.seroprevalence(model_inputs_root, verbose=verbose)
    seroprevalence_samples = sample_seroprevalence(seroprevalence, n_samples, correlate_samples, bootstrap, verbose=verbose)
    
    # ## ## ## ## ## #### ## ## ## ## ## ## ## ## ## ## ##
    # ## tweaks
    # # only take some old age from Danish blood bank data
    # age_spec_population = model_inputs.population(model_inputs_root, by_age=True)
    # pct_65_69 = age_spec_population.loc[78, 65].item() / age_spec_population.loc[78, 65:].sum()
    # danish_sub_70plus = (vaccinated.loc[[78], 'cumulative_adults_vaccinated'] + \
    #     vaccinated.loc[[78], 'cumulative_essential_vaccinated'] + \
    #     (pct_65_69 * vaccinated.loc[[78], 'cumulative_elderly_vaccinated'])).rename('cumulative_all_vaccinated')
    # vaccinated.loc[[78], 'cumulative_all_vaccinated'] = danish_sub_70plus
    
    # # above chunk not sufficient, don't pull vaccinated people out of Danish data
    # vaccinated.loc[[78]] *= 0
    ## ## ## ## ## #### ## ## ## ## ## ## ## ## ## ## ##
    
    # make pop group specific
    age_spec_population = model_inputs.population(model_inputs_root, by_age=True)
    vaccinated = get_pop_vaccinated(age_spec_population, vaccinated)
    
    # use 90% of total vaccinated
    vaccinated['vaccinated'] *= VAX_SERO_PROB
    
    if verbose:
        logger.info('Removing vaccinated from reported seroprevalence.')
    _rv = functools.partial(
        remove_vaccinated,
        vaccinated=vaccinated.copy(),
    )
    seroprevalence = remove_vaccinated(seroprevalence=seroprevalence, vaccinated=vaccinated)
    with multiprocessing.Pool(int(CONTROLLER_MP_THREADS)) as p:
        seroprevalence_samples = list(tqdm(p.imap(_rv, seroprevalence_samples), total=n_samples, file=sys.stdout))
    
    return seroprevalence, seroprevalence_samples


def get_pop_vaccinated(age_spec_population: pd.Series, vaccinated: pd.Series):
    age_spec_population = age_spec_population.reset_index()
    population = []
    for age_start in range(0, 25, 5):
        for age_end in [65, 125]:
            _population = (age_spec_population
                           .loc[(age_spec_population['age_group_years_start'] >= age_start) &
                                (age_spec_population['age_group_years_end'] <= age_end)]
                           .groupby('location_id', as_index=False)['population'].sum())
            _population['age_group_years_start'] = age_start
            _population['age_group_years_end'] = age_end
            population.append(_population)
    population = pd.concat(population)
    vaccinated = vaccinated.reset_index().merge(population)
    is_adult_only = vaccinated['age_group_years_end'] == 65
    vaccinated.loc[is_adult_only, 'vaccinated'] = vaccinated.loc[is_adult_only, ['cumulative_adults_vaccinated',
                                                                                 'cumulative_essential_vaccinated']].sum(axis=1) / \
                                                  vaccinated.loc[is_adult_only, 'population']
    vaccinated.loc[~is_adult_only, 'vaccinated'] = vaccinated.loc[~is_adult_only, 'cumulative_all_vaccinated'] / \
                                                   vaccinated.loc[~is_adult_only, 'population']
    vaccinated = vaccinated.loc[:, ['location_id', 'date', 'age_group_years_start', 'age_group_years_end', 'vaccinated']]
    
    return vaccinated


def remove_vaccinated(seroprevalence: pd.DataFrame,
                      vaccinated: pd.Series,) -> pd.DataFrame:
    seroprevalence['age_group_years_start'] = seroprevalence['study_start_age'].fillna(20)
    seroprevalence['age_group_years_start'] = np.round(seroprevalence['age_group_years_start'] / 5) * 5
    seroprevalence.loc[seroprevalence['age_group_years_start'] > 20, 'age_group_years_start'] = 20
    
    seroprevalence['age_group_years_end'] = seroprevalence['study_end_age'].fillna(125)
    seroprevalence.loc[seroprevalence['age_group_years_end'] <= 65, 'age_group_years_end'] = 65
    seroprevalence.loc[seroprevalence['age_group_years_end'] > 65, 'age_group_years_end'] = 125
    
    ## start
    # seroprevalence = seroprevalence.rename(columns={'date':'end_date'})
    # seroprevalence = seroprevalence.rename(columns={'start_date':'date'})
    ##
    ## midpoint
    seroprevalence = seroprevalence.rename(columns={'date':'end_date'})
    seroprevalence['n_midpoint_days'] = (seroprevalence['end_date'] - seroprevalence['start_date']).dt.days / 2
    seroprevalence['n_midpoint_days'] = seroprevalence['n_midpoint_days'].astype(int)
    seroprevalence['date'] = seroprevalence.apply(lambda x: x['end_date'] - pd.Timedelta(days=x['n_midpoint_days']), axis=1)
    ##
    ## always
    start_len = len(seroprevalence)
    seroprevalence = seroprevalence.merge(vaccinated, how='left')
    if len(seroprevalence) != start_len:
        raise ValueError('Sero data expanded in vax merge.')
    if seroprevalence.loc[seroprevalence['vaccinated'].isnull(), 'date'].max() >= pd.Timestamp('2020-12-01'):
        raise ValueError('Missing vax after model start (2020-12-01).')
    seroprevalence['vaccinated'] = seroprevalence['vaccinated'].fillna(0)
    ##
    ## start
    # seroprevalence = seroprevalence.rename(columns={'date':'start_date'})
    # seroprevalence = seroprevalence.rename(columns={'end_date':'date'})
    ##
    ## midpoint
    del seroprevalence['date']
    del seroprevalence['n_midpoint_days']
    seroprevalence = seroprevalence.rename(columns={'end_date':'date'})
    ##
    
    seroprevalence.loc[seroprevalence['test_target'] != 'spike', 'vaccinated'] = 0
    
    seroprevalence = seroprevalence.rename(columns={'seroprevalence':'reported_seroprevalence'})
    
    seroprevalence['seroprevalence'] = 1 - (1 - seroprevalence['reported_seroprevalence']) / (1 - seroprevalence['vaccinated'])
    
    del seroprevalence['vaccinated']
    
    return seroprevalence


def load_sensitivity(model_inputs_root: Path, n_samples: int,
                     floor: float = 1e-4, logit_se_cap: float = 1.,):
    sensitivity_data = model_inputs.assay_sensitivity(model_inputs_root)
    logit_mean = logit(sensitivity_data['sensitivity_mean'].clip(floor, 1 - floor))
    logit_sd = sensitivity_data['sensitivity_std'] / \
               (sensitivity_data['sensitivity_mean'].clip(floor, 1 - floor) * \
                (1.0 - sensitivity_data['sensitivity_mean'].clip(floor, 1 - floor)))
    logit_sd = logit_sd.clip(0, logit_se_cap)

    logit_samples = np.random.normal(loc=logit_mean.to_frame().values,
                                     scale=logit_sd.to_frame().values,
                                     size=(len(sensitivity_data), n_samples),)
    samples = expit(logit_samples)

    ## CANNOT DO THIS, MOVES SOME ABOVE 1
    # # re-center around original mean
    # samples *= sensitivity_data[['sensitivity_mean']].values / samples.mean(axis=1, keepdims=True)
    
    # sort
    samples = np.sort(samples, axis=1)

    sample_list = []
    for sample in samples.T:
        _sample = sensitivity_data.drop(['sensitivity_mean', 'sensitivity_std',], axis=1).copy()
        _sample['sensitivity'] = sample
        sample_list.append(_sample.reset_index(drop=True))
    
    return sensitivity_data, sample_list


def apply_waning_adjustment(sensitivity_data: pd.DataFrame,
                            assay_map: pd.DataFrame,
                            hospitalized_weights: pd.Series,
                            seroprevalence: pd.DataFrame,
                            daily_deaths: pd.Series,
                            pred_ifr: pd.Series,
                            durations: Dict,
                            verbose: bool = True,) -> pd.DataFrame:
    data_assays = sensitivity_data['assay'].unique().tolist()
    excluded_data_assays = [da for da in data_assays if da not in ASSAYS]
    if verbose and excluded_data_assays:
        logger.warning(f"Excluding the following assays found in sensitivity data: {', '.join(excluded_data_assays)}")
    if any([a not in data_assays for a in ASSAYS]):
        raise ValueError('Assay mis-labelled.')
    sensitivity_data = sensitivity_data.loc[sensitivity_data['assay'].isin(ASSAYS)]
    
    source_assays = sensitivity_data[['source', 'assay']].drop_duplicates().values.tolist()
    
    sensitivity = pd.concat(
        [
            fit_hospital_weighted_sensitivity_decay(
                sensitivity_data.loc[(sensitivity_data['source'] == source) & (sensitivity_data['assay'] == assay)].copy(),
                assay in INCREASING,
                hospitalized_weights.copy()
            )
            for source, assay in source_assays]
    ).set_index(['assay', 'location_id', 't']).sort_index()
    
    seroprevalence = seroprevalence.loc[seroprevalence['is_outlier'] == 0]
    
    seroprevalence = seroprevalence.merge(assay_map, how='left')
    missing_match = seroprevalence['assay_map'].isnull()
    is_N = seroprevalence['test_target'] == 'nucleocapsid'
    is_S = seroprevalence['test_target'] == 'spike'
    is_other = ~(is_N | is_S)
    seroprevalence.loc[missing_match & is_N, 'assay_map'] = 'N-Roche, N-Abbott'
    seroprevalence.loc[missing_match & is_S, 'assay_map'] = 'S-Roche, S-Ortho Ig, S-Ortho IgG, S-DiaSorin, S-EuroImmun'  # , S-Oxford
    seroprevalence.loc[missing_match & is_other, 'assay_map'] = 'N-Roche, ' \
                                                                'N-Abbott, ' \
                                                                'S-Roche, ' \
                                                                'S-Ortho Ig, S-Ortho IgG, ' \
                                                                'S-DiaSorin, S-EuroImmun'  # , S-Oxford
    if seroprevalence['assay_map'].isnull().any():
        raise ValueError(f"Unmapped seroprevalence data: {seroprevalence.loc[seroprevalence['assay_map'].isnull()]}")

    assay_combinations = seroprevalence['assay_map'].unique().tolist()
    
    infections = ((daily_deaths / pred_ifr)
                  .dropna()
                  .rename('infections')
                  .reset_index()
                  .set_index('location_id'))
    infections['date'] -= pd.Timedelta(days=durations['sero_to_death'])
    
    sensitivity_list = []
    seroprevalence_list = []
    for assay_combination in assay_combinations:
        logger.info(f'Adjusting for sensitvity decay: {assay_combination}')
        ac_sensitivity = (sensitivity
                          .loc[assay_combination.split(', ')]
                          .reset_index()
                          .groupby(['location_id', 't'])['sensitivity'].mean())
        ac_seroprevalence = (seroprevalence
                             .loc[seroprevalence['assay_map'] == assay_combination].copy())
        ac_seroprevalence = waning_adjustment(
            infections.copy(),
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


def fit_sensitivity_decay_curvefit(t: np.array, sensitivity: np.array, increasing: bool, t_N: int = 720) -> pd.DataFrame:
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


def fit_sensitivity_decay_mrbrt(sensitivity_data: pd.DataFrame, increasing: bool, t_N: int = 720) -> pd.DataFrame:
    sensitivity_data = sensitivity_data.loc[:, ['t', 'sensitivity',]]
    sensitivity_data['sensitivity'] = logit(sensitivity_data['sensitivity'])
    sensitivity_data['intercept'] = 1
    sensitivity_data['se'] = 1
    sensitivity_data['location_id'] = 1

    if increasing:
        mono_dir = 'increasing'
    else:
        mono_dir = 'decreasing'
        
    n_k = min(max(len(sensitivity_data) - 3, 2), 10,)
    k = np.hstack([[0, 0.1], np.linspace(0.1, 1, n_k)[1:]])
    max_t = sensitivity_data['t'].max()

    mr_model = mrbrt.run_mr_model(
        model_data=sensitivity_data,
        dep_var='sensitivity', dep_var_se='se',
        fe_vars=['intercept', 't'], re_vars=[],
        group_var='location_id',
        prior_dict={'intercept':{},
                    't': {'use_spline': True,
                          'spline_knots_type': 'domain',
                          'spline_knots': np.linspace(0, 1, n_k),
                          'spline_degree': 1,
                          'prior_spline_monotonicity': mono_dir,
                          'prior_spline_monotonicity_domain': (60 / max_t, 1),
                         },}
    )
    t_pred = np.arange(t_N + 1)
    sensitivity_pred, _ = mrbrt.predict(
        pred_data=pd.DataFrame({'intercept': 1,
                                't': t_pred,
                                'location_id': 1,
                                'date': t_pred,}),
        hierarchy=None,
        mr_model=mr_model,
        pred_replace_dict={},
        pred_exclude_vars=[],
        dep_var='sensitivity', dep_var_se='se',
        fe_vars=['t'], re_vars=[],
        group_var='location_id',
        sensitivity=True,
    )
        
    return pd.DataFrame({'t': t_pred, 'sensitivity': expit(sensitivity_pred['sensitivity'])})


def fit_hospital_weighted_sensitivity_decay(sensitivity: pd.DataFrame, increasing: bool,
                                            hospitalized_weights: pd.Series,) -> pd.DataFrame:
    assay = sensitivity['assay'].unique().item()
    
    source = sensitivity['source'].unique().item()
    
    if source not in ['Peluso', 'Perez-Saez', 'Bond', 'Muecksch', 'Lumley']:
        raise ValueError(f'Unexpected sensitivity source: {source}')
    
    hosp_sensitivity = sensitivity.loc[sensitivity['hospitalization_status'] == 'Hospitalized']
    nonhosp_sensitivity = sensitivity.loc[sensitivity['hospitalization_status'] == 'Non-hospitalized']
    if source == 'Peluso':
        hosp_sensitivity = fit_sensitivity_decay_curvefit(hosp_sensitivity['t'].values,
                                                          hosp_sensitivity['sensitivity'].values,
                                                          increasing,)
        nonhosp_sensitivity = fit_sensitivity_decay_curvefit(nonhosp_sensitivity['t'].values,
                                                             nonhosp_sensitivity['sensitivity'].values,
                                                             increasing,)
    else:
        hosp_sensitivity = fit_sensitivity_decay_mrbrt(hosp_sensitivity.loc[:, ['t', 'sensitivity']],
                                                       increasing,)
        nonhosp_sensitivity = fit_sensitivity_decay_mrbrt(hosp_sensitivity.loc[:, ['t', 'sensitivity']],
                                                          increasing,)
    sensitivity = (hosp_sensitivity
                   .rename(columns={'sensitivity':'hosp_sensitivity'})
                   .merge(nonhosp_sensitivity
                          .rename(columns={'sensitivity':'nonhosp_sensitivity'})))
    sensitivity['key'] = 0
    hospitalized_weights = hospitalized_weights.rename('hospitalized_weights')
    hospitalized_weights = hospitalized_weights.reset_index()
    hospitalized_weights['key'] = 0
    sensitivity = sensitivity.merge(hospitalized_weights, on='key', how='outer')
    
    sensitivity['hosp_sensitivity'] = scale_to_bounds(sensitivity['hosp_sensitivity'],
                                                      SEROREV_LB, 1.,)
    sensitivity['nonhosp_sensitivity'] = scale_to_bounds(sensitivity['nonhosp_sensitivity'],
                                                         SEROREV_LB, 1.,)
    sensitivity['sensitivity'] = (sensitivity['hosp_sensitivity'] * sensitivity['hospitalized_weights']) + \
                                 (sensitivity['nonhosp_sensitivity'] * (1 - sensitivity['hospitalized_weights']))
    sensitivity = sensitivity.reset_index()
    
    sensitivity['assay'] = assay
    
    return sensitivity.loc[:, ['location_id', 'assay', 't', 'sensitivity', 'hosp_sensitivity', 'nonhosp_sensitivity']]


def calulate_waning_factor(infections: pd.DataFrame, sensitivity: pd.DataFrame,
                           sero_date: pd.Timestamp, sero_corr: bool,) -> float:
    infections['t'] = (sero_date - infections['date']).dt.days
    infections = infections.loc[infections['t'] >= 0]
    if sero_corr not in [0, 1]:
        raise ValueError('`manufacturer_correction` should be 0 or 1.')
    if sero_corr == 1:
        # study adjusted for sensitivity, set baseline to 1
        sensitivity /= sensitivity.max()
    infections = infections.merge(sensitivity.reset_index(), how='left')
    if infections['sensitivity'].isnull().any():
        raise ValueError(f"Unmatched sero/sens points: {infections.loc[infections['sensitivity'].isnull()]}")
    waning_factor = infections['infections'].sum() / (infections['infections'] * infections['sensitivity']).sum()
    waning_factor = max(1, waning_factor)

    return waning_factor
    
    
def location_waning_adjustment(location_id: int,
                               infections: pd.DataFrame, sensitivity: pd.DataFrame,
                               seroprevalence: pd.DataFrame) -> pd.DataFrame:
    infections = infections.loc[location_id]
    sensitivity = sensitivity.loc[location_id]
    seroprevalence = seroprevalence.loc[seroprevalence['location_id'] == location_id,
                                        ['data_id', 'date', 'manufacturer_correction', 'seroprevalence']
                                       ].reset_index(drop=True)
    adj_seroprevalence = []
    for i, (sero_data_id, sero_date, sero_corr, sero_value) in enumerate(zip(seroprevalence['data_id'],
                                                                             seroprevalence['date'],
                                                                             seroprevalence['manufacturer_correction'],
                                                                             seroprevalence['seroprevalence'],)):
        waning_factor = calulate_waning_factor(infections.copy(), sensitivity.copy(),
                                               sero_date, sero_corr,)
        adj_seroprevalence.append(pd.DataFrame({
            'data_id': sero_data_id,
            'date': sero_date,
            'seroprevalence': sero_value * waning_factor
        }, index=[i]))
    adj_seroprevalence = pd.concat(adj_seroprevalence)
    adj_seroprevalence['location_id'] = location_id
    
    return adj_seroprevalence


def waning_adjustment(infections: pd.Series, sensitivity: pd.DataFrame,
                      seroprevalence: pd.DataFrame) -> pd.DataFrame:
    # # determine waning adjustment based on midpoint of survey
    # orig_date = seroprevalence[['data_id', 'date']].copy()
    # seroprevalence['n_midpoint_days'] = (seroprevalence['date'] - seroprevalence['start_date']).dt.days / 2
    # seroprevalence['n_midpoint_days'] = seroprevalence['n_midpoint_days'].astype(int)
    # seroprevalence['date'] = seroprevalence.apply(lambda x: x['date'] - pd.Timedelta(days=x['n_midpoint_days']), axis=1)
    # del seroprevalence['n_midpoint_days']
    
    seroprevalence_list = []
    location_ids = seroprevalence['location_id'].unique().tolist()
    location_ids = [location_id for location_id in location_ids if location_id in infections.reset_index()['location_id'].to_list()]
    
    _lwa = functools.partial(
        location_waning_adjustment,
        infections=infections, sensitivity=sensitivity,
        seroprevalence=seroprevalence,
    )
    with multiprocessing.Pool(int(OMP_NUM_THREADS)) as p:
        seroprevalence = list(tqdm(p.imap(_lwa, location_ids), total=len(location_ids), file=sys.stdout))
    seroprevalence = pd.concat(seroprevalence).reset_index(drop=True)
        
    # del seroprevalence['date']
    # seroprevalence = seroprevalence.merge(orig_date)
    
    return seroprevalence


def plotter(location_id: int, location_name: str,
            out_path: Path,
            seroprevalence: pd.DataFrame, ifr_results: namedtuple,
            reinfection_inflation_factor: pd.Series,
            vaccine_coverage: pd.DataFrame,
            sensitivity: pd.DataFrame,
            sensitivity_data: pd.DataFrame,
            population: pd.Series,
            **kwargs,):
    # subset location
    seroprevalence = seroprevalence.loc[seroprevalence['location_id'] == location_id]
    reinfection_inflation_factor = reinfection_inflation_factor.loc[reinfection_inflation_factor['location_id'] == location_id]
    adj_seroprevalence = ifr_results.seroprevalence.copy()
    adj_seroprevalence = adj_seroprevalence.loc[adj_seroprevalence['location_id'] == location_id]
    infections = (ifr_results.daily_numerator / ifr_results.pred).rename('infections').loc[location_id].dropna()
    infections.index -= pd.Timedelta(days=SERO_TO_DEATH)
    sensitivity = sensitivity.loc[sensitivity['location_id'] == location_id]
    vaccinated = vaccine_coverage.loc[location_id, 'cumulative_all_vaccinated'] / population.loc[location_id]

    # remove repeat infections we added earlier
    adj_seroprevalence = adj_seroprevalence.merge(reinfection_inflation_factor, how='left')
    adj_seroprevalence['inflation_factor'] = adj_seroprevalence['inflation_factor'].fillna(1)
    adj_seroprevalence['seroprevalence'] /= adj_seroprevalence['inflation_factor']
    del adj_seroprevalence['inflation_factor']

    # combine sero data
    seroprevalence = seroprevalence.rename(columns={'seroprevalence':'seroprevalence_sub_vacc'})
    seroprevalence = (seroprevalence
                      .merge(adj_seroprevalence.loc[:, ['data_id', 'seroprevalence', 'assay']],
                             how='left'))

    # make assay table
    assay_table = (seroprevalence
                   .loc[seroprevalence['is_outlier'] == 0,
                        ['data_id', 'test_name', 'test_target', 'isotype', 'assay']])
    for t_col in ['test_name', 'test_target', 'isotype']:
        assay_table[t_col] = assay_table[t_col].fillna('Not assigned')
    assay_table['assay'] = assay_table['assay'].fillna('N/A')
    assay_table = assay_table.groupby(['test_name', 'test_target', 'isotype', 'assay'])['data_id'].apply(list).reset_index()
    assay_table = assay_table.rename(columns={'test_name': 'Test label',
                                              'test_target': 'Antigen target',
                                              'isotype': 'Isotype',
                                              'assay': 'Mapped assay(s)',
                                              'data_id': 'data_id_list'})
    assay_table['Test label'] = assay_table['Test label'].apply(lambda x: text_wrap(x))
    assay_table['Mapped assay(s)'] = assay_table['Mapped assay(s)'].apply(lambda x: text_wrap(x, ', '))

    data_id_list = assay_table['data_id_list'].to_list()
    assays = assay_table['Mapped assay(s)'].to_list()
    cell_text = assay_table.values[:,:-1].tolist()
    col_labels = assay_table.columns[:-1]
    cell_colors = PLOT_C_LIST[:len(assay_table)]
    cell_colors = [[cc] * len(col_labels) for cc in cell_colors]

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])

    sero_ax = fig.add_subplot(gs[0:2, 0])

    inliers = seroprevalence.loc[seroprevalence['seroprevalence'].notnull()]
    sero_ax.scatter(inliers['date'], inliers['reported_seroprevalence'],
                    marker='s', c='none', edgecolors='black', s=100, alpha=0.5, label='Reported')
    sero_ax.scatter(inliers['date'], inliers['seroprevalence_sub_vacc'],
                    marker='^', c='none', edgecolors='black', s=100, alpha=0.5, label='No vaccinated')
    sero_ax.scatter(inliers['date'], inliers['seroprevalence'],
                    marker='o', c='none', edgecolors='black', s=100, alpha=0.5, label='No vaccinated, waning-adjusted')
    outliers = seroprevalence.loc[seroprevalence['seroprevalence'].isnull()]
    sero_ax.scatter(outliers['date'], outliers['reported_seroprevalence'],
                    marker='x', c='black', s=100, alpha=0.5, label='Outlier')

    for data_ids, c, ec in zip(data_id_list, PLOT_C_LIST, PLOT_EC_LIST):
        plot_data = seroprevalence.loc[seroprevalence['data_id'].isin(data_ids)]
        sero_ax.scatter(plot_data['date'], plot_data['reported_seroprevalence'],
                   marker='s', c='none', edgecolors=ec, s=100)
        sero_ax.scatter(plot_data['date'], plot_data['seroprevalence_sub_vacc'],
                   marker='^', c='none', edgecolors=ec, s=100)
        sero_ax.scatter(plot_data['date'], plot_data['seroprevalence'],
                   marker='o', c=c, edgecolors=ec, s=100)
    sero_ax.legend(loc=2)
    sero_ax.set_ylabel('Seroprevalence')
    sero_y_max = seroprevalence[['seroprevalence', 'seroprevalence_sub_vacc', 'reported_seroprevalence']].max(axis=1).max() * 1.05
    sero_ax.set_ylim(0, sero_y_max)
    
    infec_ax = sero_ax.twinx()
    infec_ax.plot(infections, color=PLOT_INF_C, alpha=0.5)
    infec_ax.get_yaxis().set_ticks([])
    inf_y_max = infections.max() * 1.05
    infec_ax.set_ylim(0, inf_y_max)

    sero_ax.set_xlim(PLOT_START_DATE, PLOT_END_DATE)
    sero_ax.xaxis.set_major_locator(PLOT_DATE_LOCATOR)
    sero_ax.xaxis.set_major_formatter(PLOT_DATE_FORMATTER)

    table_ax = fig.add_subplot(gs[2, :])
    table_ax.axis('tight')
    table_ax.axis('off')
    table = table_ax.table(cellText=cell_text,
                     cellColours=cell_colors,
                     colLabels=col_labels,
                     colWidths=[0.3, 0.1, 0.1, 0.3],
                     cellLoc='left',
                     loc='center',)
    table.scale(1, 3)
    table.set_fontsize(20)

    sens_ax = fig.add_subplot(gs[0, 1])
    for i, (assay, c) in enumerate(zip(assays, PLOT_C_LIST)):
        assay_sensitivity = sensitivity.loc[sensitivity['assay'] == assay.replace('\n', '')]
        if assay_sensitivity.empty:
            raise ValueError(f'Unable to find sensitivity curve for {assay}')
        sens_ax.plot(assay_sensitivity['t'],
                     assay_sensitivity['sensitivity'],
                     color=c, linewidth=2)
        for a in assay.split(', '):
            sens_ax.scatter(sensitivity_data.loc[sensitivity_data['assay'] == a, 't'],
                            sensitivity_data.loc[sensitivity_data['assay'] == a, 'sensitivity'],
                            marker='.', color=c, s=100, alpha=0.25)
    sens_ax.axvline((infections.index.max() -  PLOT_START_DATE).days,
                    linestyle='--', color='darkgrey',)
    sens_ax.set_ylim(0, 1.05)
    sens_ax.set_ylabel('Sensitivity')
    sens_ax.set_xlabel('Time from exposure to test')

    vacc_ax = fig.add_subplot(gs[1, 1])
    vacc_ax.plot(vaccinated * 100, color='black')
    vacc_ax.set_ylabel('Vaccinated (%)')
    vacc_ax.set_xlim(PLOT_START_DATE, PLOT_END_DATE)
    vacc_ax.xaxis.set_major_locator(PLOT_DATE_LOCATOR)
    vacc_ax.xaxis.set_major_formatter(PLOT_DATE_FORMATTER)

    fig.suptitle(f'{location_name} ({location_id})', fontsize=24)
    if out_path is None:
        fig.show()
    else:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
