from pathlib import Path
from typing import Tuple
from loguru import logger

import pandas as pd
import numpy as np

from covid_historical_model.etl import db, helpers


def seroprevalence(model_inputs_root: Path, verbose: bool = True,) -> pd.DataFrame:
    # load
    data_path = model_inputs_root / 'serology' / 'global_serology_summary.csv'
    data = pd.read_csv(data_path)
    if verbose:
        logger.info(f'Initial observation count: {len(data)}')

    # date formatting
    for date_var in ['start_date', 'date']:
        # data[date_var] = helpers.str_fmt(data[date_var]).replace('.202$', '.2020')
        # data.loc[(data['location_id'] == 570) & (data[date_var] == '11.08.2021'), date_var] = '11.08.2020'
        # data.loc[(data['location_id'] == 533) & (data[date_var] == '13.11.2.2020'), date_var] = '13.11.2020'
        # data.loc[data[date_var] == '05.21.2020', date_var] = '21.05.2020'
        data[date_var] = pd.to_datetime(data[date_var], format='%d.%m.%Y')
    
    # if no start date provided, assume 2 weeks before end date? median seems to be 21 days, so this is conservative
    data['start_date'] = data['start_date'].fillna(data['date'] - pd.Timedelta(days=14))

    # convert to m/l/u to 0-1, sample size to numeric
    if not (helpers.str_fmt(data['units']).unique() == 'percentage').all():
        raise ValueError('Units other than percentage present.')
    data['lower'] = helpers.str_fmt(data['lower']).replace('not specified', np.nan).astype(float)
    data['upper'] = helpers.str_fmt(data['upper']).replace('not specified', np.nan).astype(float)
    data['seroprevalence'] = data['value'] / 100
    data['seroprevalence_lower'] = data['lower'] / 100
    data['seroprevalence_upper'] = data['upper'] / 100
    data['sample_size'] = helpers.str_fmt(data['sample_size']).replace(('unchecked', 'not specified'), np.nan).astype(float)
    
    data['bias'] = helpers.str_fmt(data['bias']).replace(('unchecked', 'not specified'), np.nan).astype(float)
    
    outliers = []
    data['manual_outlier'] = data['manual_outlier'].fillna(0)
    manual_outlier = data['manual_outlier']
    outliers.append(manual_outlier)
    if verbose:
        logger.info(f'{manual_outlier.sum()} rows from sero data flagged as outliers in ETL.')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## SOME THINGS
    # # 1)
    # #    Question: How to get complete SS?
    # #    Current approach: CI -> SE -> SS where possible; fill with min(SS) where we also don't have CI (very few rows).
    # #    Final solution: ...
    # ss = ss_from_ci(data['seroprevalence'], data['seroprevalence_lower'], data['seroprevalence_upper'])
    # n_missing_ss = (data['sample_size'].isnull() & ss.notnull()).sum()
    # n_missing_ss_ci = (data['sample_size'].isnull() & ss.isnull()).sum()
    # data['sample_size'] = data['sample_size'].fillna(ss)
    # data['sample_size'] = data['sample_size'].fillna(data['sample_size'].min())
    # logger.info(f'Inferring sample size from CI for {n_missing_ss} studies; '
    #             f'filling missing sample size with min observed for {n_missing_ss_ci} that also do not report CI.')
    # del n_missing_ss, n_missing_ss_ci
    
    # # 2)
    # #    Question: What if survey is only in adults? Only kids?
    # #    Current approach: Drop beyond some threshold limits.
    # #    Final solution: ...
    # max_start_age = 30
    # min_end_age = 60
    # data['study_start_age'] = helpers.str_fmt(data['study_start_age']).replace('not specified', np.nan).astype(float)
    # data['study_end_age'] = helpers.str_fmt(data['study_end_age']).replace('not specified', np.nan).astype(float)
    # too_old = data['study_start_age'] > max_start_age
    # too_young = data['study_end_age'] < min_end_age
    # age_outlier = (too_old  | too_young).astype(int)
    # outliers.append(age_outlier)
    # if verbose:
    #     logger.info(f'{age_outlier.sum()} rows from sero data do not have enough '
    #             f'age coverage (at least ages {max_start_age} to {min_end_age}).')
    
    # 3)
    #    Question: Use of geo_accordance?
    #    Current approach: Drop non-represeentative (geo_accordance == 0).
    #    Final solution: ...
    data['geo_accordance'] = helpers.str_fmt(data['geo_accordance']).replace(('unchecked', np.nan), '0').astype(int)
    geo_outlier = data['geo_accordance'] == 0
    outliers.append(geo_outlier)
    if verbose:
        logger.info(f'{geo_outlier.sum()} rows from sero data do not have `geo_accordance`.')
    data['correction_status'] = helpers.str_fmt(data['correction_status']).replace(('unchecked', 'not specified', np.nan), '0').astype(int)
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## manually specify certain tests when reporting is mixed (might just be US?)
    # Connecticut (looks to use Abbott test)
    is_conn = data['location_id'] == 529
    is_cdc = data['survey_series'] == 'cdc_series'
    data.loc[is_conn & is_cdc, 'test_target'] = 'nucleocapsid'
    data.loc[is_conn & is_cdc, 'isotype'] = 'IgG'
    data.loc[is_conn & is_cdc, 'test_name'] = 'Abbott ARCHITECT SARS-CoV-2 IgG immunoassay'
    
    # Illinois (looks to use Ortho test)
    is_conn = data['location_id'] == 536
    is_cdc = data['survey_series'] == 'cdc_series'
    data.loc[is_conn & is_cdc, 'test_target'] = 'spike'
    data.loc[is_conn & is_cdc, 'isotype'] = 'IgG'
    data.loc[is_conn & is_cdc, 'test_name'] = 'Ortho-Clinical Diagnostics VITROS SARS-CoV-2 IgG immunoassay'
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    keep_columns = ['data_id', 'nid', 'location_id', 'start_date', 'date',
                    'seroprevalence',  # 'sample_size',
                    'test_name', 'test_target', 'isotype',
                    'bias', 'bias_type',
                    'correction_status', 'geo_accordance',
                    'is_outlier', 'manual_outlier']
    data['is_outlier'] = pd.concat(outliers, axis=1).max(axis=1).astype(int)
    data = (data
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))
    data['data_id'] = data.index
    data = data.loc[:, keep_columns]
    
    if verbose:
        logger.info(f"Final inlier count: {len(data.loc[data['is_outlier'] == 0])}")
        logger.info(f"Final outlier count: {len(data.loc[data['is_outlier'] == 1])}")
    
    return data


def reported_epi(model_inputs_root: Path, input_measure: str,
                 hierarchy: pd.DataFrame, em_path: Path = None,) -> Tuple[pd.Series, pd.Series]:
    data_path = model_inputs_root / 'use_at_your_own_risk' / 'full_data_extra_hospital.csv'
    data = pd.read_csv(data_path)
    data = data.rename(columns={'Deaths':'cumulative_deaths',
                                'Confirmed':'cumulative_cases',
                                'Hospitalizations':'cumulative_hospitalizations',})
    data['date'] = pd.to_datetime(data['Date'])
    keep_cols = ['location_id', 'date', f'cumulative_{input_measure}']
    data = data.loc[:, keep_cols].dropna()
    data['location_id'] = data['location_id'].astype(int)
    
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: helpers.fill_dates(x, [f'cumulative_{input_measure}']))
            .reset_index(drop=True))
    if input_measure == 'deaths' and em_path is not None:
        em_data = pd.read_csv(em_path)
        em_data = em_data.rename(columns={'value':'em_scalar'})
        em_data = em_data.loc[:, ['location_id', 'em_scalar']]
        data = data.merge(em_data, how='left')
        data['em_scalar'] = data['em_scalar'].fillna(1)
        data['em_scalar'] = data['em_scalar'].clip(1, np.inf)
        data[f'cumulative_{input_measure}'] *= data['em_scalar']
        del data['em_scalar']
    data = helpers.aggregate_data_from_md(data, hierarchy, f'cumulative_{input_measure}')
    data[f'daily_{input_measure}'] = (data
                                      .groupby('location_id')[f'cumulative_{input_measure}']
                                      .apply(lambda x: x.diff())
                                      .fillna(data[f'cumulative_{input_measure}']))
    data = data.dropna()
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index())
    
    cumulative_data = data[f'cumulative_{input_measure}']
    daily_data = data[f'daily_{input_measure}']

    return cumulative_data, daily_data


def hierarchy(model_inputs_root:Path, covariates: bool = False) -> pd.DataFrame:
    if not covariates:
        data_path = model_inputs_root / 'locations' / 'modeling_hierarchy.csv'
    else:
        data_path = model_inputs_root / 'locations' / 'covariate_with_aggregates_hierarchy.csv'
    
    data = pd.read_csv(data_path)
    data = data.sort_values('sort_order').reset_index(drop=True)
    
    return data


def population(model_inputs_root: Path, by_age: bool = False) -> pd.Series:
    data_path = model_inputs_root / 'output_measures' / 'population' / 'all_populations.csv'
    data = pd.read_csv(data_path)
    is_2019 = data['year_id'] == 2019
    is_bothsex = data['sex_id'] == 3
    if by_age:
        age_metadata = db.age_metadata()    
        data = (data
                .loc[is_2019 & is_bothsex, ['location_id', 'age_group_id', 'population']])
        data = data.merge(age_metadata)
        data = (data
                .set_index(['location_id', 'age_group_years_start', 'age_group_years_end'])
                .sort_index()
                .loc[:, 'population'])
    else:
        is_allage = data['age_group_id'] == 22
        data = (data
                .loc[is_2019 & is_bothsex & is_allage]
                .set_index('location_id')
                .sort_index()
                .loc[:, 'population'])

    return data


def assay_sensitivity(model_inputs_root: Path, assay_day_0: int = 21,) -> pd.DataFrame:
    # TODO: bootstrapping or something to incorporate uncertainty (would need to digitize this portion from Perez-Saez plots)?
    peluso_path = model_inputs_root / 'serology' / 'waning_immunity' / 'peluso_assay_sensitivity.xlsx'
    perez_saez_paths = [
        model_inputs_root / 'serology' / 'waning_immunity' / 'perez-saez_n-roche.xlsx',
        model_inputs_root / 'serology' / 'waning_immunity' / 'perez-saez_rbd-euroimmun.xlsx',
        model_inputs_root / 'serology' / 'waning_immunity' / 'perez-saez_rbd-roche.xlsx',
    ]
    
    peluso = pd.read_excel(peluso_path)
    peluso['t'] = peluso['Time'].apply(lambda x: int(x.split(' ')[0]) * 30)
    peluso = peluso.rename(columns={'mean': 'sensitivity',
                                    'AntigenAndAssay': 'assay',
                                    'Hospitalization_status': 'hospitalization_status',})
    peluso = peluso.loc[:, ['assay', 'hospitalization_status', 't', 'sensitivity']]
    # only need to keep commercial assays
    peluso = peluso.loc[~peluso['assay'].isin(['Neut-Monogram', 'RBD-LIPS', 'RBD-Split Luc',
                                               'RBD-Lum', 'S-Lum', 'N(full)-Lum', 'N-LIPS',
                                               'N(frag)-Lum', 'N-Split Luc'])]
    peluso['source'] = 'Peluso'
    
    # could try to manually apply hosp/non-hosp split
    perez_saez = pd.concat([pd.read_excel(perez_saez_path) for perez_saez_path in perez_saez_paths])
    perez_saez = perez_saez.loc[perez_saez['t'] >= assay_day_0]
    perez_saez['t'] -= assay_day_0
    perez_saez = pd.concat([
        pd.concat([perez_saez, pd.DataFrame({'hospitalization_status':'Non-hospitalized'}, index=perez_saez.index)], axis=1),
        pd.concat([perez_saez, pd.DataFrame({'hospitalization_status':'Hospitalized'}, index=perez_saez.index)], axis=1)
    ])
    perez_saez['source'] = 'Perez-Saez'
    
    sensitivity = pd.concat([peluso, perez_saez]).reset_index(drop=True)
    
    return sensitivity
