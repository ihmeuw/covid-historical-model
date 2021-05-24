from pathlib import Path
from loguru import logger
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns

from covid_historical_model.durations.durations import SERO_TO_DEATH, EXPOSURE_TO_SEROPOSITIVE, EXPOSURE_TO_DEATH
from covid_historical_model.etl import model_inputs
from covid_historical_model.utils.misc import text_wrap
from covid_historical_model.utils.math import logit, expit

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# should have module for these that is more robust to additions
ASSAYS = ['N-Abbott',  # IgG
          'S-Roche', 'N-Roche',  # Ig
          'S-Ortho Ig', 'S-Ortho IgG', # Ig/IgG
          'S-DiaSorin',  # IgG
          'S-EuroImmun',]  # IgG
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
PLOT_END_DATE = pd.Timestamp('2021-08-01')


def sample_seroprevalence(seroprevalence: pd.DataFrame, n_samples: int,
                          correlate_samples: bool,
                          min_samples: int = 10,
                          floor: float = 1e-5, logit_se_cap: float = 1.,
                          verbose: bool = True):
    logit_se_from_ci = lambda x: (logit(x['seroprevalence_upper']) - logit(x['seroprevalence_lower'])) / 3.92
    logit_se_from_ss = lambda x: np.sqrt((x['seroprevalence'] * (1 - x['seroprevalence'])) / x['sample_size']) /\
                                 (x['seroprevalence'] * (1.0 - x['seroprevalence']))
    if n_samples >= min_samples:
        if verbose:
            logger.info(f'Producing {n_samples} seroprevalence samples.')
        if (seroprevalence['seroprevalence'] < seroprevalence['seroprevalence_lower']).any():
            mean_sub_low = seroprevalence['seroprevalence'] < seroprevalence['seroprevalence_lower']
            raise ValueError(f"Mean seroprevalence below lower:\n{seroprevalence[mean_sub_low]}")
        if (seroprevalence['seroprevalence'] > seroprevalence['seroprevalence_upper']).any():
            high_sub_mean = seroprevalence['seroprevalence'] > seroprevalence['seroprevalence_upper']
            raise ValueError(f"Mean seroprevalence above upper:\n{seroprevalence[high_sub_mean]}")
            
        summary_vars = ['seroprevalence', 'seroprevalence_lower', 'seroprevalence_upper']
        seroprevalence[summary_vars] = seroprevalence[summary_vars].clip(floor, 1 - floor)
            
        logit_mean = logit(seroprevalence['seroprevalence'].copy())
        logit_se = logit_se_from_ci(seroprevalence.copy())
        logit_se = logit_se.fillna(logit_se_from_ss(seroprevalence.copy()))
        logit_se = logit_se.fillna(logit_se_cap)
        logit_se = logit_se.clip(0, logit_se_cap)
        logit_samples = np.random.normal(loc=logit_mean.to_frame().values,
                                         scale=logit_se.to_frame().values,
                                         size=(len(seroprevalence), n_samples))
        samples = expit(logit_samples)
        
        # re-center around original mean
        samples *= seroprevalence[['seroprevalence']].values / samples.mean(axis=1, keepdims=True)
        if correlate_samples:
            logger.info('Correlating seroprevalence samples.')
            samples = np.sort(samples, axis=1)

        seroprevalence = seroprevalence.drop(['seroprevalence', 'seroprevalence_lower', 'seroprevalence_upper', 'sample_size'],
                                             axis=1)
        bootstrap_list = []
        for sample in samples.T:
            bootstrap = seroprevalence.copy()
            bootstrap['seroprevalence'] = sample
            bootstrap_list.append(bootstrap)
    elif n_samples > 1:
        raise ValueError(f'If sampling, need at least {min_samples}.')
    else:
        if verbose:
            logger.info('Just using mean seroprevalence.')
            
        seroprevalence['seroprevalence'] = seroprevalence['seroprevalence'].clip(floor, 1 - floor)
            
        seroprevalence = seroprevalence.drop(['seroprevalence_lower', 'seroprevalence_upper', 'sample_size'],
                                             axis=1)
        bootstrap_list = [seroprevalence]
    
    return bootstrap_list


def load_seroprevalence_sub_vacccinated(model_inputs_root: Path, vaccinated: pd.Series,
                                        n_samples: int, correlate_samples: bool,
                                        verbose: bool = True) -> pd.DataFrame:
    seroprevalence = model_inputs.seroprevalence(model_inputs_root, verbose=verbose)
    seroprevalence_samples = sample_seroprevalence(seroprevalence, n_samples, correlate_samples, verbose=verbose)
    
    # ## ## ## ## ## #### ## ## ## ## ## ## ## ## ## ## ##
    # ## tweaks
    # # only take some old age from Danish blood bank data
    # age_spec_population = model_inputs.population(model_inputs_root, by_age=True)
    # pct_65_69 = age_spec_population.loc[78, 65].item() / age_spec_population.loc[78, 65:].sum()
    # danish_sub_70plus = (vaccinated.loc[[78], 'cumulative_adults_vaccinated'] + \
    #     vaccinated.loc[[78], 'cumulative_essential_vaccinated'] + \
    #     (pct_65_69 * vaccinated.loc[[78], 'cumulative_elderly_vaccinated'])).rename('cumulative_all_vaccinated')
    # vaccinated.loc[[78], 'cumulative_all_vaccinated'] = danish_sub_70plus
    # ## ## ## ## ## #### ## ## ## ## ## ## ## ## ## ## ##
    
    # use 80% of total vaccinated (effective not sufficient)
    population = model_inputs.population(model_inputs_root)
    vaccinated = vaccinated['cumulative_all_vaccinated'].rename('vaccinated') * 0.8
    vaccinated /= population
    
    # above chunk not sufficient, don't pull vaccinated people out of Danish data
    vaccinated.loc[[78]] *= 0
    
    # vaccinated = vaccinated.reset_index()
    # vaccinated['date'] += pd.Timedelta(days=EXPOSURE_TO_SEROPOSITIVE)
    # vaccinated = vaccinated.set_index(['location_id', 'date'])
    
    if verbose:
        logger.info('Removing (effectively?) vaccinated from reported seroprevalence.')
    seroprevalence_samples = [remove_vaccinated(sample, vaccinated,) for sample in seroprevalence_samples]
    
    return seroprevalence, seroprevalence_samples


def remove_vaccinated(seroprevalence: pd.DataFrame,
                      vaccinated: pd.Series,) -> pd.DataFrame:
    # ## start
    # # seroprevalence = seroprevalence.rename(columns={'date':'end_date'})
    # # seroprevalence = seroprevalence.rename(columns={'start_date':'date'})
    # ##
    # ## midpoint
    # seroprevalence = seroprevalence.rename(columns={'date':'end_date'})
    # seroprevalence['n_midpoint_days'] = (seroprevalence['end_date'] - seroprevalence['start_date']).dt.days / 2
    # seroprevalence['n_midpoint_days'] = seroprevalence['n_midpoint_days'].astype(int)
    # seroprevalence['date'] = seroprevalence.apply(lambda x: x['end_date'] - pd.Timedelta(days=x['n_midpoint_days']), axis=1)
    # ##
    ## always
    seroprevalence = seroprevalence.merge(vaccinated.reset_index(), how='left')
    # ##
    # ## start
    # # seroprevalence = seroprevalence.rename(columns={'date':'start_date'})
    # # seroprevalence = seroprevalence.rename(columns={'end_date':'date'})
    # ##
    # ## midpoint
    # del seroprevalence['date']
    # del seroprevalence['n_midpoint_days']
    # seroprevalence = seroprevalence.rename(columns={'end_date':'date'})
    # ##
    seroprevalence['vaccinated'] = seroprevalence['vaccinated'].fillna(0)
    
    seroprevalence.loc[seroprevalence['test_target'] != 'spike', 'vaccinated'] = 0
    
    seroprevalence = seroprevalence.rename(columns={'seroprevalence':'reported_seroprevalence'})
    
    seroprevalence['seroprevalence'] = 1 - (1 - seroprevalence['reported_seroprevalence']) / (1 - seroprevalence['vaccinated'])
    
    del seroprevalence['vaccinated']
    
    return seroprevalence


def apply_waning_adjustment(sensitivity: pd.DataFrame,
                            hospitalized_weights: pd.Series,
                            seroprevalence: pd.DataFrame,
                            daily_deaths: pd.Series,
                            pred_ifr: pd.Series,
                            verbose: bool = True,) -> pd.DataFrame:
    data_assays = sensitivity['assay'].unique().tolist()
    excluded_data_assays = [da for da in data_assays if da not in ASSAYS]
    if verbose and excluded_data_assays:
        logger.warning(f"Excluding the following assays found in sensitivity data: {', '.join(excluded_data_assays)}")
    if any([a not in data_assays for a in ASSAYS]):
        raise ValueError('Assay mis-labelled.')
    sensitivity = sensitivity.loc[sensitivity['assay'].isin(ASSAYS)]
    
    source_assays = sensitivity[['source', 'assay']].drop_duplicates().values.tolist()
    
    sensitivity = pd.concat(
        [
            fit_hospital_weighted_sensitivity_decay(
                sensitivity.loc[(sensitivity['source'] == source) & (sensitivity['assay'] == assay)].copy(),
                assay in INCREASING,
                hospitalized_weights.copy()
            )
            for source, assay in source_assays]
    ).set_index(['assay', 'location_id', 't']).sort_index()
    
    seroprevalence = seroprevalence.loc[seroprevalence['is_outlier'] == 0]
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## DO THIS DIFFERENTLY...?
    assay_map = pd.read_excel('/'.join(__file__.split('/')[:-2]) + '/maps/assay_map.xlsx')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    
    seroprevalence = seroprevalence.merge(assay_map, how='left')
    missing_match = seroprevalence['assay_map'].isnull()
    is_N = seroprevalence['test_target'] == 'nucleocapsid'
    is_S = seroprevalence['test_target'] == 'spike'
    is_other = ~(is_N | is_S)
    seroprevalence.loc[missing_match & is_N, 'assay_map'] = 'N-Roche, N-Abbott'
    seroprevalence.loc[missing_match & is_S, 'assay_map'] = 'S-Roche, S-Ortho Ig, S-Ortho IgG, S-DiaSorin, S-EuroImmun'
    seroprevalence.loc[missing_match & is_other, 'assay_map'] = 'N-Roche, ' \
                                                                'N-Abbott, ' \
                                                                'S-Roche, S-Ortho Ig, ' \
                                                                'S-Ortho IgG, S-DiaSorin, S-EuroImmun' 
    if seroprevalence['assay_map'].isnull().any():
        raise ValueError(f"Unmapped seroprevalence data: {seroprevalence.loc[seroprevalence['assay_map'].isnull()]}")

    assay_combinations = seroprevalence['assay_map'].unique().tolist()

    sensitivity_list = []
    seroprevalence_list = []
    for assay_combination in assay_combinations:
        ac_sensitivity = (sensitivity
                             .loc[assay_combination.split(', ')]
                             .reset_index()
                             .groupby(['location_id', 't'])['sensitivity'].mean())
        ac_seroprevalence = (seroprevalence
                             .loc[seroprevalence['assay_map'] == assay_combination].copy())
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
    infections = infections.loc[infections['t'] >= 0]
    infections = infections.merge(sensitivity.reset_index(), how='left')
    if infections['sensitivity'].isnull().any():
        raise ValueError(f"Unmatched sero/sens points: {infections.loc[infections['sensitivity'].isnull()]}")
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
    
    # # determine waning adjustment based on midpoint of survey
    # orig_date = seroprevalence[['data_id', 'date']].copy()
    # seroprevalence['n_midpoint_days'] = (seroprevalence['date'] - seroprevalence['start_date']).dt.days / 2
    # seroprevalence['n_midpoint_days'] = seroprevalence['n_midpoint_days'].astype(int)
    # seroprevalence['date'] = seroprevalence.apply(lambda x: x['date'] - pd.Timedelta(days=x['n_midpoint_days']), axis=1)
    # del seroprevalence['n_midpoint_days']
    
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
    effectively_vaccinated = vaccine_coverage.loc[location_id, 'cumulative_all_effective'] / population.loc[location_id]

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
    vacc_ax.plot(effectively_vaccinated * 100, color='black')
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
